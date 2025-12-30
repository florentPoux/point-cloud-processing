import numpy as np

def export_to_ifc(bim_model, filepath, project_name="Scanned Building"):
    """Export BIM model to IFC format"""
    try:
        import ifcopenshell
        from ifcopenshell.api import run
    except ImportError:
        return export_simple_ifc_text(bim_model, filepath, project_name)

    ifc_file = ifcopenshell.file(schema="IFC4")

    project = run("root.create_entity", ifc_file, ifc_class="IfcProject", name=project_name)
    site = run("root.create_entity", ifc_file, ifc_class="IfcSite", name="Site")
    building = run("root.create_entity", ifc_file, ifc_class="IfcBuilding", name="Building")
    storey = run("root.create_entity", ifc_file, ifc_class="IfcBuildingStorey", name="Ground Floor")

    run("aggregate.assign_object", ifc_file, relating_object=project, product=site)
    run("aggregate.assign_object", ifc_file, relating_object=site, product=building)
    run("aggregate.assign_object", ifc_file, relating_object=building, product=storey)

    for i, wall_geom in enumerate(bim_model['walls']):
        wall = run("root.create_entity", ifc_file, ifc_class="IfcWall", name=f"Wall_{i}")
        run("spatial.assign_container", ifc_file, relating_structure=storey, product=wall)

    for i, door_geom in enumerate(bim_model['doors']):
        door = run("root.create_entity", ifc_file, ifc_class="IfcDoor", name=f"Door_{i}")
        run("spatial.assign_container", ifc_file, relating_structure=storey, product=door)

    for i, window_geom in enumerate(bim_model['windows']):
        window = run("root.create_entity", ifc_file, ifc_class="IfcWindow", name=f"Window_{i}")
        run("spatial.assign_container", ifc_file, relating_structure=storey, product=window)

    for i, column_geom in enumerate(bim_model['columns']):
        column = run("root.create_entity", ifc_file, ifc_class="IfcColumn", name=f"Column_{i}")
        run("spatial.assign_container", ifc_file, relating_structure=storey, product=column)

    ifc_file.write(filepath)

    return True

def export_simple_ifc_text(bim_model, filepath, project_name):
    """Export simple IFC text file"""
    with open(filepath, 'w') as f:
        f.write("ISO-10303-21;\n")
        f.write("HEADER;\n")
        f.write(f"FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\n")
        f.write(f"FILE_NAME('{filepath}','2025-01-01T00:00:00',(''),(''),'IfcOpenShell','IfcOpenShell','');\n")
        f.write("FILE_SCHEMA(('IFC4'));\n")
        f.write("ENDSEC;\n")
        f.write("DATA;\n")

        entity_id = 1

        f.write(f"#{entity_id}=IFCPROJECT('{project_name}','','',$,$,$,$,$,$);\n")
        entity_id += 1

        for i, wall in enumerate(bim_model['walls']):
            f.write(f"#{entity_id}=IFCWALL('Wall_{i}','','',$,$,$,$,$);\n")
            entity_id += 1

        f.write("ENDSEC;\n")
        f.write("END-ISO-10303-21;\n")

    return True

def create_ifc_metadata(bim_model):
    """Create IFC metadata dictionary"""
    return {
        'schema': 'IFC4',
        'application': 'Point Cloud Processing Library',
        'timestamp': '2025-01-01T00:00:00',
        'author': '3D Geodata Academy',
        'organization': 'Smart Point Cloud',
        'elements': bim_model['metadata']
    }
