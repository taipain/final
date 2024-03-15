from icrawler.builtin import GoogleImageCrawler
import torch
def download_images(keyword, max_num):
    google_crawler = GoogleImageCrawler(storage={"root_dir": keyword})
    google_crawler.crawl(keyword=keyword, max_num=max_num)

search = '川上洋平'
download_images(search, 100)
