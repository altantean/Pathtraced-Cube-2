struct obj;

struct obj : vertloader<obj>
{
    obj(const char *name) : vertloader(name) {}

    static const char *formatname() { return "obj"; }
    static bool animated() { return false; }
    bool flipy() const { return true; }
    int type() const { return MDL_OBJ; }

    struct objmeshgroup : vertmeshgroup
    {
        void parsevert(char *s, vector<vec> &out)
        {
            vec &v = out.add(vec(0, 0, 0));
            while(isalpha(*s)) s++;
            loopi(3)
            {
                v[i] = strtod(s, &s);
                while(isspace(*s)) s++;
                if(!*s) break;
            }
        }

        struct objtri { int vert[3]; };

        void flushgroup(const char *mname, vector<vert> &verts, vector<tcvert> &tcverts, vector<objtri> &tris, bool hasnormals, float smooth)
        {
            if(tris.empty()) return;
            if(!hasnormals)
            {
                vertmesh helper;  // the normal smoothing lives on mesh, any instance will do
                helper.group = this;
                if(smooth <= 1) helper.mesh::smoothnorms(verts.getbuf(), verts.length(), tris.getbuf(), tris.length(), smooth, true);
                else helper.mesh::buildnorms(verts.getbuf(), verts.length(), tris.getbuf(), tris.length(), true);
            }
            const int MAXMESHVERTS = 0xFFFF;
            int *remap = new int[verts.length()];
            memset(remap, -1, verts.length()*sizeof(int));
            vector<int> used;  // this piece's vertices (group indices), in local order
            vector<tri> piece;
            #define EMITPIECE do {                 vertmesh &m = *new vertmesh;                 m.group = this;                 m.name = mname[0] ? newstring(mname) : NULL;                 meshes.add(&m);                 m.numverts = used.length();                 m.verts = new vert[used.length()];                 m.tcverts = new tcvert[used.length()];                 loopvk(used) { m.verts[k] = verts[used[k]]; m.tcverts[k] = tcverts[used[k]]; remap[used[k]] = -1; }                 m.numtris = piece.length();                 m.tris = new tri[piece.length()];                 memcpy(m.tris, piece.getbuf(), piece.length()*sizeof(tri));                 used.setsize(0);                 piece.setsize(0);             } while(0)
            loopv(tris)
            {
                const objtri &t = tris[i];
                int fresh = 0;
                loopk(3) if(remap[t.vert[k]] < 0) fresh++;
                if(used.length() + fresh > MAXMESHVERTS) EMITPIECE;
                tri &lt = piece.add();
                loopk(3)
                {
                    int &r = remap[t.vert[k]];
                    if(r < 0) { r = used.length(); used.add(t.vert[k]); }
                    lt.vert[k] = ushort(r);
                }
            }
            if(piece.length()) EMITPIECE;
            #undef EMITPIECE
            delete[] remap;
        }

        bool load(const char *filename, float smooth)
        {
            int len = strlen(filename);
            if(len < 4 || strcasecmp(&filename[len-4], ".obj")) return false;

            stream *file = openfile(filename, "rb");
            if(!file) return false;
            const int loadstart = SDL_GetTicks();

            name = newstring(filename);

            numframes = 1;

            vector<vec> attrib[3];
            char buf[512];

            hashtable<ivec, int> verthash;
            vector<vert> verts;
            vector<tcvert> tcverts;
            vector<objtri> tris;

            #define STARTMESH do {                 ingroup = true;                 copystring(groupname, meshname);                 verthash.clear();                 verts.setsize(0);                 tcverts.setsize(0);                 tris.setsize(0);             } while(0)

            #define FLUSHMESH do {                 flushgroup(groupname, verts, tcverts, tris, !attrib[2].empty(), smooth);                 ingroup = false;             } while(0)

            string meshname = "", groupname = "";
            bool ingroup = false;
            while(file->getline(buf, sizeof(buf)))
            {
                char *c = buf;
                while(isspace(*c)) c++;
                switch(*c)
                {
                    case '#': continue;
                    case 'v':
                        if(isspace(c[1])) parsevert(c, attrib[0]);
                        else if(c[1]=='t') parsevert(c, attrib[1]);
                        else if(c[1]=='n') parsevert(c, attrib[2]);
                        break;
                    case 'g':
                    {
                        while(isalpha(*c)) c++;
                        while(isspace(*c)) c++;
                        char *name = c;
                        size_t namelen = strlen(name);
                        while(namelen > 0 && isspace(name[namelen-1])) namelen--;
                        copystring(meshname, name, min(namelen+1, sizeof(meshname)));

                        if(ingroup) FLUSHMESH;
                        break;
                    }
                    case 'f':
                    {
                        if(!ingroup) STARTMESH;
                        int v0 = -1, v1 = -1;
                        while(isalpha(*c)) c++;
                        for(;;)
                        {
                            while(isspace(*c)) c++;
                            if(!*c) break; 
                            ivec vkey(-1, -1, -1);
                            loopi(3)
                            {
                                vkey[i] = strtol(c, &c, 10);
                                if(vkey[i] < 0) vkey[i] = attrib[i].length() + vkey[i];
                                else vkey[i]--;
                                if(!attrib[i].inrange(vkey[i])) vkey[i] = -1;
                                if(*c!='/') break;
                                c++;
                            }
                            int *index = verthash.access(vkey);
                            if(!index)
                            {
                                index = &verthash[vkey];
                                *index = verts.length();
                                vert &v = verts.add();
                                v.pos = vkey.x < 0 ? vec(0, 0, 0) : attrib[0][vkey.x];
                                v.pos = vec(v.pos.z, -v.pos.x, v.pos.y);
                                v.norm = vkey.z < 0 ? vec(0, 0, 0) : attrib[2][vkey.z];
                                v.norm = vec(v.norm.z, -v.norm.x, v.norm.y);
                                tcvert &tcv = tcverts.add();
                                tcv.tc = vkey.y < 0 ? vec2(0, 0) : vec2(attrib[1][vkey.y].x, 1-attrib[1][vkey.y].y);
                            }
                            if(v0 < 0) v0 = *index;
                            else if(v1 < 0) v1 = *index;
                            else
                            {
                                objtri &t = tris.add();
                                t.vert[0] = *index;
                                t.vert[1] = v1;
                                t.vert[2] = v0;
                                v1 = *index;
                            }
                        }
                        break;
                    }
                }
            }

            if(ingroup) FLUSHMESH;
            #undef STARTMESH
            #undef FLUSHMESH

            delete file;

            int totaltris = 0, totalverts = 0;
            loopv(meshes) { totaltris += ((vertmesh *)meshes[i])->numtris; totalverts += ((vertmesh *)meshes[i])->numverts; }
            if(totalverts > 0xFFFF) conoutf("large obj %s: %d triangles, %d vertices in %d meshes (%d ms)", filename, totaltris, totalverts, meshes.length(), int(SDL_GetTicks() - loadstart));

            return true;
        }
    };

    meshgroup *loadmeshes(const char *name, va_list args)
    {
        objmeshgroup *group = new objmeshgroup;
        if(!group->load(name, va_arg(args, double))) { delete group; return NULL; }
        return group;
    }

    bool loaddefaultparts()
    {
        part &mdl = addpart();
        const char *pname = parentdir(name);
        defformatstring(name1, "packages/models/%s/tris.obj", name);
        mdl.meshes = sharemeshes(path(name1), 2.0);
        if(!mdl.meshes)
        {
            defformatstring(name2, "packages/models/%s/tris.obj", pname);    // try obj in parent folder (vert sharing)
            mdl.meshes = sharemeshes(path(name2), 2.0);
            if(!mdl.meshes) return false;
        }
        Texture *tex, *masks;
        loadskin(name, pname, tex, masks);
        mdl.initskins(tex, masks);
        if(tex==notexture) conoutf(CON_ERROR, "could not load model skin for %s", name1);
        return true;
    }
};

vertcommands<obj> objcommands;

