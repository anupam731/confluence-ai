#!/usr/bin/env python3

"""
XML Structure Debugger - Find out why streaming fails
"""

import xml.etree.ElementTree as ET
from pathlib import Path

def debug_xml_structure(xml_file: str):
    print(f"🔍 DEBUGGING: {xml_file}")
    print("=" * 50)
    
    # Method 1: Your working approach (ET.parse)
    print("\n1️⃣ TESTING YOUR WORKING METHOD (ET.parse):")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        
        print(f"✅ Root tag: {root.tag}")
        print(f"✅ Namespace: '{ns}'")
        
        # Count objects
        page_count = 0
        content_count = 0
        current_pages = 0
        
        for obj in root.findall(f'.//{ns}object'):
            obj_class = obj.get('class')
            if obj_class == 'Page':
                page_count += 1
                # Check if current
                for child in obj:
                    if child.tag == f'{ns}property' and child.get('name') == 'contentStatus':
                        if child.text == 'current':
                            current_pages += 1
                        break
            elif obj_class == 'BodyContent':
                content_count += 1
        
        print(f"✅ Total Page objects: {page_count}")
        print(f"✅ Current pages: {current_pages}")
        print(f"✅ BodyContent objects: {content_count}")
        
        # Show first current page details
        print(f"\n🔍 First current page details:")
        for obj in root.findall(f'.//{ns}object'):
            if obj.get('class') == 'Page':
                page_id = None
                title = None
                status = None
                
                for child in obj:
                    if child.tag == f'{ns}id' and child.get('name') == 'id':
                        page_id = child.text
                    elif child.tag == f'{ns}property':
                        name = child.get('name')
                        if name == 'contentStatus':
                            status = child.text
                        elif name == 'title':
                            title = child.text
                
                if status == 'current':
                    print(f"   ID: {page_id}")
                    print(f"   Title: {title}")
                    print(f"   Status: {status}")
                    break
        
    except Exception as e:
        print(f"❌ ET.parse failed: {e}")
    
    # Method 2: Streaming approach (ET.iterparse)
    print(f"\n2️⃣ TESTING STREAMING METHOD (ET.iterparse):")
    try:
        context = ET.iterparse(xml_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        
        print(f"✅ Root tag: {root.tag}")
        print(f"✅ Namespace: '{ns}'")
        
        object_count = 0
        page_objects = 0
        current_pages = 0
        
        current_element = None
        
        for event, elem in context:
            if event == 'start':
                if elem.tag == f'{ns}object':
                    current_element = elem
            elif event == 'end':
                if elem.tag == f'{ns}object' and current_element is not None:
                    object_count += 1
                    obj_class = current_element.get('class')
                    
                    if obj_class == 'Page':
                        page_objects += 1
                        
                        # Extract data IMMEDIATELY
                        page_id = None
                        status = None
                        title = None
                        
                        for child in current_element:
                            if child.tag == f'{ns}id' and child.get('name') == 'id':
                                page_id = child.text
                            elif child.tag == f'{ns}property':
                                name = child.get('name')
                                text_value = child.text
                                if name == 'contentStatus':
                                    status = text_value
                                elif name == 'title':
                                    title = text_value
                        
                        if status == 'current':
                            current_pages += 1
                            if current_pages == 1:  # Show first current page
                                print(f"🔍 First current page found:")
                                print(f"   ID: {page_id}")
                                print(f"   Title: {title}")
                                print(f"   Status: {status}")
                    
                    # Clear after processing
                    current_element.clear()
                    current_element = None
                
                # Stop after reasonable number for debugging
                if object_count >= 1000:
                    break
        
        print(f"✅ Objects processed: {object_count}")
        print(f"✅ Page objects: {page_objects}")
        print(f"✅ Current pages: {current_pages}")
        
    except Exception as e:
        print(f"❌ ET.iterparse failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("🎯 COMPARISON COMPLETE")

if __name__ == "__main__":
    import sys
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/entities.xml"
    debug_xml_structure(xml_file)
