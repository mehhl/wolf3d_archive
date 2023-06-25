_ADDON_DIR = os.path.join(os.getcwd(), '..', 'assets')
_ADDON_DIR
floor_size = 10
remove_ball = True
# it is enough for me to specify an absolute path to our 'assets' dir
# what to set nsmap to? it has something to do with namespaces
# check what is the nsmap attribute of an existing _Element
print(mjcf.nsmap)
print(xml_tools.find_element(mjcf, 'skin', 'skin').nsmap)
help(etree._Element.makeelement)
# see if the height field is there
os.listdir(_ADDON_DIR)
os.path.join(_ADDON_DIR, 'heightmap.png')

# construction of assets object experiments
from dm_control.suite import common
assets = common.ASSETS.copy()
type(assets)
assets.keys()
help(resources)
help(os.walk)
type(xml_string)
type(mjcf)



    



"""
    xml_string, assets = dog.get_model_and_assets(floor_size, remove_ball)
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    addons = {}
    _, _, filenames = next(resources.WalkResources(_ADDON_DIR))
    for filename in filenames:
        addons[filename] = resources.GetResource(os.path.join(_ADDON_DIR, filename))

    if 'heightmap.png' not in addons:
        warnings.warn("File heightmap.png not found in {}".format(_ADDON_DIR))
    if 'skin_texture.png' not in addons:
        warnings.warn("File skin_texture.png not found in {}".format(_ADDON_DIR))
    if 'dog_skin.skn' not in addons:
        warnings.warn("File dog_skin.skn not found in {}".format(_ADDON_DIR))
    assets.update(addons)

    if 'heightmap.png' in addons:
        xml_assets = mjcf.find('asset')
        hfield_id = 'heightfield'
        hfield = mjcf.makeelement('hfield',
                                  {"name": hfield_id,
                                  "file": 'heightmap.png',
                                  "size": "50 50 0 1"})
        xml_assets.append(hfield)
        
        worldbody = mjcf.find('worldbody')
        geom_hfield = mjcf.makeelement('geom',
                                      {'name': 'geom_hfield',
                                       'type': 'hfield',
                                       'hfield': hfield_id,
                                       'quat': '0 0 0 1'})
        worldbody.replace(xml_tools.find_element(worldbody, 'geom', 'floor'),
                          geom_hfield)

"""
    # camera work
    
    # add seekers
    add_seekers_to_model()  


