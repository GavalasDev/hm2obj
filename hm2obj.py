#!/usr/bin/env python3 

from PIL import Image, ImageFilter
import numpy as np

import argparse

class HeightMap2OBJ:
    def __init__(self, file_path, blur=None, downsample_step=10, mask=None, min_z=0, max_z=100, px_per_km=100):
        self.downsample=downsample_step
        self.im = Image.open(file_path, 'r')
        if blur:
            self.im = self.im.filter(filter=ImageFilter.GaussianBlur(blur))
        self.im = np.asarray(self.im)[:, :, 0].copy()

        self.width, self.height = self.im.shape

        self.im = np.interp(self.im, (np.min(self.im), np.max(self.im)), (min_z, max_z))
        self.m_per_px = 1000 / px_per_km

        if mask:
            self.mask = Image.open(mask, 'r')
            self.mask = np.asarray(self.mask)[::self.downsample, ::self.downsample, 0].copy().flatten()
            self.mask = self.mask != 0
        else:
            self.mask = None


    def calc_verts(self):
        vertices = []

        for x in range(0, self.width, self.downsample):
            for y in range(0, self.height, self.downsample):
                z = self.im[x, y]
                vertices.append([y*self.m_per_px, z, x*self.m_per_px])

        return np.array(vertices)

    def calc_faces(self):
        # simple grid
        faces = []

        size_x = len(range(0, self.width, self.downsample))
        size_y = len(range(0, self.height, self.downsample))

        for i in range(1, (size_x-1) * size_y):
            if i % size_y == 0:
                continue
            faces.append((i, i+1, i+size_y))
            faces.append((i+1, i+1+size_y, i+size_y))

        return np.array(faces)


    def apply_mask(self, vertices, faces): 
        verts = vertices[self.mask]

        dropped = np.where(~self.mask)[0] + 1

        m = np.logical_not(np.any(np.isin(faces, dropped), axis=1))
        facs = faces[m]
        
        indices, counts = np.unique(facs.flatten(), return_counts=True)
        edge_indices = indices[counts < 6] - 1

        vertices[edge_indices, 1] = 0

        # return all vertices or the indices of facs are wrong
        return (vertices, facs)


    def vertex_string(self, x, y, z):
        return f"v {x} {y} {z}"

    def face_string(self, i, j, k):
        return f"f {i}/{i}/1 {j}/{j}/1 {k}/{k}/1"

    def uv_string(self, u, v):
        return f"vt {u} {v}"


    def generate_obj(self, output=None):
        vertices = self.calc_verts()
        faces = self.calc_faces()

        if self.mask is not None:
            vertices, faces = self.apply_mask(vertices, faces)
    
        file = open(output, 'w') if output else None

        for v in vertices:
            print(self.vertex_string(v[0], v[1], v[2]), file=file)

        print("", file=file)

        for v in vertices:
            print(self.uv_string(v[0]/(self.height*self.m_per_px), -v[2]/(self.width*self.m_per_px)), file=file)

        print("", file=file)

        print("vn 0 1 0\n", file=file)

        print("s 1\n", file=file)

        print("", file=file)

        for f in faces:
            print(self.face_string(f[2], f[1], f[0]), file=file) # fixed orientation



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='HM2OBJ',
                    description='Converts black and white heightmap images to obj files.',
                    epilog='Example Usage\n\n\t./hm2obj.py -d 10 --minz=0 --maxz=567 --px_per_km=260 -m examples/mask.png -o santorini.obj examples/heightmap.png')

    parser.add_argument('filename') 
    parser.add_argument('-o', '--output') 
    parser.add_argument('-b', '--blur', type=int, default=None, help='Radius to be used for Gaussian blur filter') 
    parser.add_argument('-d', '--downsample', type=int, default=1, help='Calculates the z value for every d pixels') 
    parser.add_argument('-m', '--mask', default=None, help='Image file to be used as mask') 
    parser.add_argument('--minz', type=int, default=0, help='Minimum value for z') 
    parser.add_argument('--maxz', type=int, default=100, help='Maximum value for z') 
    parser.add_argument('--px_per_km', type=int, default=500, help='Number of pixels equivalent to 1 km') 

    args = parser.parse_args()

    HeightMap2OBJ(args.filename, blur=args.blur, downsample_step=args.downsample,mask=args.mask, min_z=args.minz, max_z=args.maxz, px_per_km=args.px_per_km).generate_obj(output=args.output)
