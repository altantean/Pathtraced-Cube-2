#ifndef RT_WATER_SPECTRUM_GLSL
#define RT_WATER_SPECTRUM_GLSL

// energy density S(k) (height^2 per unit wavenumber), deep-water dispersion
float jonswapK(float k, float g, float U, float F) {
    if (k <= 0.0) return 0.0;
    float omega = sqrt(g * k);
    float wp = 22.0 * pow(g * g / (U * F), 1.0 / 3.0);  // peak angular frequency
    float alpha = 0.076 * pow(U * U / (F * g), 0.22);  // Phillips constant
    float sigma = omega <= wp ? 0.07 : 0.09;
    float r = exp(-(omega - wp) * (omega - wp) / (2.0 * sigma * sigma * wp * wp));
    float q = wp / omega;
    float Sw = alpha * g * g / (omega * omega * omega * omega * omega) * exp(-1.25 * q * q * q * q) * pow(3.3, r);
    return Sw * g / (2.0 * omega);  // S(omega) * d(omega)/dk
}

// depth limitation
float tmaFactor(float k, float g, float depth) {
    float wh = sqrt(k * max(depth, 0.0));  // omega * sqrt(depth / g), omega = sqrt(g k)
    if (wh <= 1.0) return 0.5 * wh * wh;
    if (wh < 2.0) return 1.0 - 0.5 * (2.0 - wh) * (2.0 - wh);
    return 1.0;
}
#endif  // RT_WATER_SPECTRUM_GLSL
