#!/usr/bin/env python3

"""
Smart Hybrid Confluence XML Parser
Automatically chooses ET.parse() or iterparse() based on file size
"""

import xml.etree.ElementTree as ET
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
import logging
from typing import List, Dict, Any, Iterator
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartHybridParser:
    """Hybrid parser that chooses optimal method based on file size"""
    
    def __init__(self, batch_size: int = 32, output_dir: str = "./data/parsed", size_threshold_gb: float = 1.0):
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.size_threshold_gb = size_threshold_gb
        self.pages_processed = 0
        self.batches_written = 0
        self.parsing_method = None
    
    def clean_confluence_content(self, raw_content: str) -> str:
        """Clean and normalize Confluence content"""
        if not raw_content:
            return ""
        
        content = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', raw_content, flags=re.DOTALL)
        content = re.sub(r'<\?xml[^>]*\?>', '', content)
        content = re.sub(r'<ac:structured-macro[^>]*>.*?</ac:structured-macro>', '', content, flags=re.DOTALL)
        content = re.sub(r'<ri:attachment[^>]*/?>', '', content)
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator="\n")
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def extract_attachments_from_body(self, body_html: str) -> List[str]:
        """Extract attachment filenames from body HTML"""
        attachments = []
        if not body_html:
            return attachments
        
        soup = BeautifulSoup(body_html, 'html.parser')
        for attachment in soup.find_all('ri:attachment'):
            filename = attachment.get('ri:filename')
            if filename and not filename.lower().startswith('screenshot'):
                attachments.append(filename)
        
        return attachments
    
    def is_current_version(self, obj, ns: str) -> bool:
        """Check if a page object is current version"""
        has_original_version = False
        has_space = False
        
        for child in obj:
            if child.tag == f'{ns}property':
                name = child.get('name')
                if name == 'originalVersion':
                    has_original_version = True
                elif name == 'space':
                    has_space = True
        
        return not has_original_version and has_space
    
    def parse_with_full_load(self, xml_file: str) -> Iterator[List[Dict[str, Any]]]:
        """Parse using ET.parse() for smaller files"""
        logger.info("Using ET.parse() method (full load) - optimal for smaller files")
        self.parsing_method = "ET.parse"
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        
        logger.info(f"ns: '{ns}'")
        
        # Same logic as your working parser
        all_pages = {}
        contents = {}
        
        # First pass: collect Page objects
        for obj in root.findall(f'.//{ns}object'):
            if obj.get('class') == 'Page':
                page_id = None
                page_data = {}
                content_status = None
                
                for child in obj:
                    if child.tag == f'{ns}id' and child.get('name') == 'id':
                        page_id = child.text
                    elif child.tag == f'{ns}property':
                        name = child.get('name')
                        if name == 'contentStatus':
                            content_status = child.text
                        elif name in ['title', 'space', 'version', 'creator', 'creationDate', 'lastModificationDate']:
                            page_data[name] = child.text or ""
                
                if page_id and content_status == 'current':
                    page_data['is_current_version'] = self.is_current_version(obj, ns)
                    all_pages[page_id] = page_data
        
        # Second pass: collect BodyContent objects
        for obj in root.findall(f'.//{ns}object'):
            if obj.get('class') == 'BodyContent':
                content_id = None
                body = ""
                
                for child in obj:
                    if child.tag == f'{ns}property':
                        name = child.get('name')
                        if name == 'body':
                            body = child.text or ""
                        elif name == 'content':
                            for subchild in child:
                                if subchild.tag == f'{ns}id' and subchild.get('name') == 'id':
                                    content_id = subchild.text
                
                if content_id:
                    contents[content_id] = body
        
        logger.info(f"Full parse complete: {len(all_pages)} current pages, {len(contents)} content objects")
        
        # Filter and generate batches
        final_pages = self._filter_current_versions(all_pages)
        yield from self._generate_batches(final_pages, contents)
    
    def parse_with_streaming(self, xml_file: str) -> Iterator[List[Dict[str, Any]]]:
        """Parse using iterparse() for larger files with proper memory management"""
        logger.info("Using iterparse() method (streaming) - optimal for larger files")
        self.parsing_method = "iterparse"
        
        all_pages = {}
        contents = {}
        
        # Use iterparse with proper memory management (from search results)
        context = ET.iterparse(xml_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        
        logger.info(f"ns: '{ns}'")
        
        current_element = None
        elements_processed = 0
        
        for event, elem in context:
            if event == 'start':
                if elem.tag == f'{ns}object':
                    current_element = elem
            
            elif event == 'end':
                if elem.tag == f'{ns}object' and current_element is not None:
                    obj_class = current_element.get('class')
                    elements_processed += 1
                    
                    if obj_class == 'Page':
                        page_id, page_data = self._extract_page_data_streaming(current_element, ns)
                        if page_id and page_data:
                            all_pages[page_id] = page_data
                    
                    elif obj_class == 'BodyContent':
                        content_id, body = self._extract_content_data_streaming(current_element, ns)
                        if content_id and body:
                            contents[content_id] = body
                    
                    if elements_processed % 1000 == 0:
                        logger.info(f"Processed {elements_processed} objects (Current pages: {len(all_pages)}, Content: {len(contents)})")
                    
                    # Critical: Clear element immediately after processing (from search results)
                    current_element.clear()
                    current_element = None
                
                # Clear non-object elements to save memory
                elif elem.tag != f'{ns}object':
                    elem.clear()
        
        # Clear root to free memory
        root.clear()
        
        logger.info(f"Streaming parse complete: {len(all_pages)} current pages, {len(contents)} content objects")
        
        # Filter and generate batches
        final_pages = self._filter_current_versions(all_pages)
        yield from self._generate_batches(final_pages, contents)
    
    def _extract_page_data_streaming(self, elem, ns: str) -> tuple:
        """Extract page data for streaming method"""
        page_id = None
        page_data = {}
        content_status = None
        
        # Extract data immediately before any clearing
        for child in elem:
            if child.tag == f'{ns}id' and child.get('name') == 'id':
                page_id = child.text
            elif child.tag == f'{ns}property':
                name = child.get('name')
                value = child.text if child.text is not None else ""
                
                if name == 'contentStatus':
                    content_status = value
                elif name in ['title', 'space', 'version', 'creator', 'creationDate', 'lastModificationDate']:
                    page_data[name] = value
        
        if page_id and content_status == 'current':
            page_data['is_current_version'] = self.is_current_version(elem, ns)
            return page_id, page_data
        
        return None, None
    
    def _extract_content_data_streaming(self, elem, ns: str) -> tuple:
        """Extract content data for streaming method"""
        content_id = None
        body = ""
        
        for child in elem:
            if child.tag == f'{ns}property':
                name = child.get('name')
                if name == 'body':
                    body = child.text if child.text is not None else ""
                elif name == 'content':
                    for subchild in child:
                        if subchild.tag == f'{ns}id' and subchild.get('name') == 'id':
                            content_id = subchild.text
        
        if content_id:
            return content_id, body
        
        return None, None
    
    def _filter_current_versions(self, all_pages: Dict) -> Dict:
        """Filter to current versions - same logic for both methods"""
        pages_by_title = defaultdict(list)
        
        for page_id, page_data in all_pages.items():
            title = page_data.get('title', '')
            pages_by_title[title].append((page_id, page_data))
        
        final_pages = {}
        for title, page_list in pages_by_title.items():
            if len(page_list) == 1:
                page_id, page_data = page_list[0]
                if page_data['is_current_version']:
                    final_pages[page_id] = page_data
            else:
                current_candidates = [(pid, pdata) for pid, pdata in page_list if pdata['is_current_version']]
                if current_candidates:
                    def sort_key(item):
                        pid, pdata = item
                        version = int(pdata.get('version', '0'))
                        date = pdata.get('lastModificationDate', '')
                        return (version, date)
                    
                    current_candidates.sort(key=sort_key, reverse=True)
                    page_id, page_data = current_candidates[0]
                    final_pages[page_id] = page_data
                    logger.info(f"Multiple versions for '{title}': selected page {page_id} (version {page_data.get('version', 'unknown')})")
        
        return final_pages
    
    def _generate_batches(self, final_pages: Dict, contents: Dict) -> Iterator[List[Dict[str, Any]]]:
        """Generate batches - same for both methods"""
        batch = []
        
        for page_id, meta in final_pages.items():
            raw_body = contents.get(page_id, "")
            cleaned_body = self.clean_confluence_content(raw_body)
            
            if cleaned_body and len(cleaned_body) > 50:
                attachments = self.extract_attachments_from_body(raw_body)
                
                record = {
                    "id": page_id,
                    "title": meta.get("title", ""),
                    "space": meta.get("space", ""),
                    "version": meta.get("version", ""),
                    "creator": meta.get("creator", ""),
                    "creationDate": meta.get("creationDate", ""),
                    "lastModificationDate": meta.get("lastModificationDate", ""),
                    "content": cleaned_body,
                    "attachments": attachments,
                    "word_count": len(cleaned_body.split())
                }
                
                batch.append(record)
                self.pages_processed += 1
                
                if len(batch) >= self.batch_size:
                    yield batch
                    self.batches_written += 1
                    logger.info(f"Generated batch {self.batches_written} ({self.pages_processed} pages total)")
                    batch = []
        
        if batch:
            yield batch
            self.batches_written += 1
            logger.info(f"Final batch {self.batches_written} ({self.pages_processed} pages total)")
    
    def parse_and_save_batches(self, xml_file: str, output_prefix: str = "confluence_batch") -> List[str]:
        """Parse XML and save pages in batched JSON files with consistent naming"""
        xml_path = Path(xml_file)
        file_size_gb = xml_path.stat().st_size / (1024**3)
        
        logger.info(f"File size: {file_size_gb:.2f} GB")
        logger.info(f"Size threshold: {self.size_threshold_gb} GB")
        
        # Choose parsing method based on file size
        if file_size_gb < self.size_threshold_gb:
            parse_method = self.parse_with_full_load
        else:
            parse_method = self.parse_with_streaming
        
        # Parse and save batches with consistent naming
        batch_files = []
        
        for batch_num, batch in enumerate(parse_method(xml_file)):
            batch_filename = f"{output_prefix}_{batch_num:04d}.json"
            batch_path = self.output_dir / batch_filename
            
            with open(batch_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, indent=2, ensure_ascii=False)
            
            batch_files.append(str(batch_path))
            logger.info(f"Saved batch {batch_num} to {batch_path} ({len(batch)} pages)")
        
        # Save manifest with parsing method info
        manifest = {
            'total_batches': len(batch_files),
            'total_pages': self.pages_processed,
            'batch_size': self.batch_size,
            'parsing_method': self.parsing_method,
            'file_size_gb': file_size_gb,
            'size_threshold_gb': self.size_threshold_gb,
            'batch_files': batch_files
        }
        
        manifest_path = self.output_dir / f"{output_prefix}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Parsing complete: {self.pages_processed} pages in {len(batch_files)} batches using {self.parsing_method}")
        return batch_files

def main():
    parser = argparse.ArgumentParser(description='Smart hybrid Confluence XML parser')
    parser.add_argument('xml_file', nargs='?', default='data/raw/entities.xml')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='./data/parsed')
    parser.add_argument('--output-prefix', default='confluence_batch')
    parser.add_argument('--size-threshold', type=float, default=1.0,
                       help='File size threshold in GB (default: 1.0)')
    
    args = parser.parse_args()
    
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return 1
    
    logger.info(f"Processing XML file: {xml_path}")
    
    # Initialize hybrid parser
    hybrid_parser = SmartHybridParser(
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        size_threshold_gb=args.size_threshold
    )
    
    try:
        batch_files = hybrid_parser.parse_and_save_batches(str(xml_path), args.output_prefix)
        
        print(f"\n✅ Smart hybrid parsing complete!")
        print(f"📊 Processed {hybrid_parser.pages_processed} pages")
        print(f"📁 Created {len(batch_files)} batch files")
        print(f"🔧 Method used: {hybrid_parser.parsing_method}")
        print(f"💾 Output directory: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
