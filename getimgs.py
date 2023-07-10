import requests
from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor
import glob
import re
import random
import os
import shutil
from PIL import Image
from lxml import html

class GetImgs:
    
    def __init__(self, object, num=1000, dir=None, overwrite=False) -> None:
        self.object = object
        self.num = num
        self.branches = []
        self.hrefs = []
        self.scraped = 0
        self.dir = dir if dir else 'images'
        self.dir = os.path.join(os.getcwd(), self.dir)
        self.session = requests.Session()
        self.overwrite = overwrite
        self.header = {
            'authority': 'www.google.com',
            'method': 'GET',
            'scheme': 'https',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-bitness': '"64"',
            'sec-ch-ua-full-version-list': '"Not.A/Brand";v="8.0.0.0", "Chromium";v="114.0.5735.199", "Google Chrome";v="114.0.5735.199"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"15.0.0"',
            'sec-ch-ua-wow64': '?0',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        

    def _extract_image_links(self, text: str) -> list:
        hrefs = re.findall(r'https://.*?"', text)
        hrefs = [href[:-1] for href in hrefs if 'https://encrypted-tbn0.gstatic.com' not in href and '.jpg' in href or '.png' in href or '.jpeg' in href or '.gif' in href or '.webp' in href]
        hrefs = list(set(hrefs))

        return hrefs
    
    def _get_google_images(self, url: str):
        headers = self.header
        response = self.session.get(url, headers=headers)
        if len(self.branches) <= 10000:
            tree = html.fromstring(response.content)
            branches = tree.xpath('//*[@id="i10"]/div[1]/span/span/div/a')
            branches = [branch.get('href') for branch in branches]
            self.branches.extend(branches)
        
        soup = bs(response.content, 'html.parser')
        scripts = soup.find_all('script')
        scripts = [script for script in scripts if 'AF_initDataCallback' in script.text]
        
        return self._extract_image_links(str(scripts[-1])), response.text
    
    def _get_batchexecute_images(self, text: str, reqid: int):
            url = 'https://www.google.com/_/VisualFrontendUi/data/batchexecute'
            sid = text.split('FdrFJe":"')[1].split('"')[0]
            form_start = text.split('null,null,null,null,null,1,[],null,null,null,')[1].split('],null,null,null,null,[')[0] + ']'
            form_data = f'[[["HoAMBc","[null,null,{form_start},null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[\\"{object}\\",null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[]],null,null,null,null,null,null,null,null,[null,\\"CAE=\\",\\"GGggAA==\\"]]",null,"generic"]]]'

            payload = {'f.req': form_data}
            params = {
                'rpcids': 'HoAMBc',
                'source-path': '/search',
                'f.sid': sid,
                'bl': 'boq_visualfrontendserver_20230629.04_p0',
                'hl': 'en-US',
                'authuser': '',
                'soc-app': '162',
                'soc-platform': '1',
                'soc-device': '1',
                '_reqid': f'{reqid}',
                'rt': 'c'
            }

            response = self.session.post(url, params=params, data=payload, headers=self.header)
            hrefs = self._extract_image_links(response.text)
            
            return hrefs

    def _get_image_batch(self, url: str):
        reqid = random.randint(10000, 99999)
        req_count = 0
        duplicate_perc = 0
        hrefs, text = self._get_google_images(url)

        while len(hrefs) <= self.num and duplicate_perc != 100:
            if req_count != 0: reqid = f'{req_count}{reqid}'
            batchexecute_hrefs = self._get_batchexecute_images(text, int(reqid)) # type: ignore
            duplicate_perc = len([href for href in batchexecute_hrefs if href in hrefs]) / len(batchexecute_hrefs) * 100 if len(batchexecute_hrefs) != 0 else 100
            hrefs.extend(batchexecute_hrefs)
            hrefs = list(set(hrefs))
            
            req_count += 1
        
        return hrefs
            
    def get_images(self, url: str):
        hrefs = []
        while len(hrefs) <= self.num:
            batch_hrefs = self._get_image_batch(url)
            hrefs.extend(batch_hrefs)
            hrefs = [href for href in hrefs if href not in self.hrefs]
            hrefs = list(set(hrefs))
            url = f'https://google.com/{self.branches.pop(0)}'
            
            self.scraped += len(batch_hrefs)
            print(f'Scraped: {len(batch_hrefs)}\t|\tTotal Scraped: {self.scraped}\t|\tUnique: {len(list(set(self.hrefs))) + len(hrefs)}')
        
        self.hrefs.extend(hrefs)
        return hrefs
        
    
    def download_image(self, image_url: str, directory: str):  
        filename = image_url.split("/")[-1]
        filename = filename.split("?")[0].replace('\\', '').replace('/', '')
        filename += '.jpg' if True not in [filename.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.gif', '.webp']] else ''
        filename = os.path.join(directory, filename)
        r = requests.get(image_url, stream = True, timeout=16)

        if r.status_code == 200:
            r.raw.decode_content = True
            
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        return filename
            
    def store_images(self, image_urls: list, directory: str):
        with ThreadPoolExecutor() as executor:
            futures = []
            for href in image_urls:
                future = executor.submit(self.download_image, href, directory)
                futures.append(future)
            
            for future in futures:
                try:
                    future.result()
                except:
                    pass
        
        return len(os.listdir(directory))
                
    
    @staticmethod
    def is_image_corrupted(image_path):
        try:
            # Attempt to open the image
            with Image.open(image_path) as img:
                img.verify()  # Verify if the image is valid
            return False  # Image is not corrupted
        except (IOError, SyntaxError):
            return True  # Image is corrupted or unreadable

    def remove_corrupted_images(self, directory, skip=0):
        num = 0
        for i, file in enumerate(sorted(glob.glob(directory + '/*'), key= lambda t: os.stat(t).st_mtime)):
            if i < skip: continue
            file_size = os.path.getsize(file)  # Get file size in bytes

            # Check if the file is a corrupted image or below 10 KB
            if self.is_image_corrupted(file) or file_size < 10240:
                os.remove(file)
                num += 1
        return num

    # Verifies whether the required number of images are successfully downloaded, continues to download if not
    def _verify_integrity(self, directory: str):
        num_images = len(os.listdir(directory))
        initial_num = self.num
        skip = 0
        
        while num_images < initial_num:
            self.num = initial_num - num_images
            hrefs = self.get_images(f'https://google.com/{self.branches.pop(0)}')
            self._pipeline(hrefs, directory, skip)
            skip = num_images
            num_images = len(os.listdir(directory))
            
        num = self.remove_corrupted_images(directory)
        print(f'{num} found to be corrupted in final check, {num_images} images successfully stored in {directory}...')
            
        # if num_images > self.num:
        #     for image in os.listdir(directory)[self.num:]:
        #         os.remove(os.path.join(directory, image))
                
    def _pipeline(self, hrefs, directory, prev_idx=0):
        stored = self.store_images(hrefs, directory)
        deleted = self.remove_corrupted_images(directory, prev_idx)
        print(f'\033[32mStored: {stored}\t|\t\033[31mCorrupted: {deleted}\033[37m')

                
    def _single(self):
        
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if self.object in os.listdir(self.dir) and self.overwrite == False:
            if len(os.listdir(os.path.join(self.dir, self.object))) >= self.num:
                print(f'{self.object} already exists in {self.dir}...')
                return
        
        self.object.replace(' ', '+')
        hrefs = self.get_images(f'https://www.google.com/search?q={self.object}&tbm=isch')

        dir = os.path.join(self.dir, self.object)
        if not os.path.exists(dir):
            os.makedirs(dir)

        self._pipeline(hrefs, dir)
        self._verify_integrity(dir)
    
    def _multiple(self):
        for object in self.object:
            GetImgs(object, num=self.num)._single()
    
    def run(self):
        if type(self.object) == str:
            return self._single()
        else:
            return self._multiple()
    
if __name__ == '__main__':
    GetImgs(['pen'], num=3200).run()