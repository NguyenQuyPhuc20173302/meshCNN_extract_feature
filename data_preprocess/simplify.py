import bpy
import os

def normalize(obj_file, target_edges, export_name):
    mesh = load_obj(obj_file)
    simplify(mesh, target_edges)
    export_obj(mesh, export_name)


def load_obj(obj_file):
    while bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)
    bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_edges=True,
                                use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
    ob = bpy.context.selected_objects[0]
    print(obj_file + "original: " + str(len(ob.data.edges)))
    return ob

def subsurf(mesh):
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide()
    bpy.ops.object.editmode_toggle()

def simplify(mesh, target_edges):
    bpy.context.view_layer.objects.active = mesh
    nedges = len(mesh.data.edges)
    
    while nedges < target_edges:
        subsurf(mesh)
        nedges = len(mesh.data.edges)
    ratio = target_edges / float(nedges)
    ratio = float('%s' % ('%.3g' % (ratio)))
    while (int(len(mesh.data.edges) * ratio) > target_edges):
        ratio = ratio - 0.001
    while (int(len(mesh.data.edges) * ratio) < target_edges):
        ratio = ratio + 0.001
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.decimate(ratio=ratio)
    bpy.ops.object.editmode_toggle()

def export_obj(mesh, export_name):
    outpath = os.path.dirname(export_name)
    if not os.path.isdir(outpath): os.makedirs(outpath)
    print('EXPORTING', export_name)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.export_mesh.stl(filepath=export_name, check_existing=False, filter_glob="*.stl")
    # bpy.ops.export_scene.obj(filepath=export_name, check_existing=False, filter_glob="*.obj;*.mtl",
    #                         use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True,
    #                         use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True,
    #                         use_uvs=False, use_materials=False, use_triangles=True, use_nurbs=False,
    #                         use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
    #                         group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO',
    #                         axis_forward='-Z', axis_up='Y')
    # with open(export_name, 'w') as f:
    #     f.write("OBJ File: \n")
    #     for v in mesh.data.vertices:
    #         f.write("v %.4f %.4f %.4f \n" % v.co[:])
    #     for p in mesh.data.polygons:
    #         f.write("\nf ")
    #         for i in p.vertices:
    #             f.write("%d " % (i + 1))

if __name__ == "__main__":
    # normalize('../dataset/raw_data/train/sphere/untitled16.obj', 750, '../dataset/clean_data/train/sphere/16.stl')
    data_dir = '..\\dataset\\raw_data'
    for dirname in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dirname)
        for subdir_name in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir_name)
            i = 0
            for filename in os.listdir(subdir_path):
                i = i + 1
                path = os.path.join(subdir_path, filename)
                obj_file = path
                target_edges = 750
                destination_path = path.split("\\")
                destination_path[2] = 'clean_data'
                destination_path[-1] = str(i) + ".stl"
                destination_path = "/".join(destination_path)
                print(destination_path)
                export_name = destination_path
                print('args: ', obj_file, target_edges, export_name)
                normalize(obj_file, target_edges, export_name)
