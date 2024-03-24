# -*-encoding:utf-8-*-
import os
import random
import xml
import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
from osgeo import gdal, gdalconst
from osgeo.osr import CoordinateTransformation, SpatialReference

def getSRSPair(dataset):
    """
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    """
    prosrs = SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def lonlat2geo(dataset, lon, lat):
    """
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    """
    prosrs, geosrs = getSRSPair(dataset)
    ct = CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def geo2imagexy(dataset, x, y):
    """
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    """
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


def read_xml(xml_file):
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    labels = []
    for idx, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        # boxes[idx, :] = [x1, y1, x1, y2, x2, y1, x2, y2]
        boxes[idx, :] = [x1, y1, x2, y2]
        labels.append("ship")
    return boxes, labels


def read_vif(vif_path=None, img_path=None):
    """
    返回一张图上所有多边形标注框的各点坐标及标注框对应标签
    """
    gdal.AllRegister()
    dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    dom = xml.dom.minidom.parse(vif_path)
    root = dom.documentElement
    children = root.getElementsByTagName('Child')
    objects_points, objects_label = [], []
    print(len(children), 'objects')

    for child in children:
        if len(child.getElementsByTagName('GeoShape')) == 0:
            continue
        geoShape = child.getElementsByTagName('GeoShape')[0]
        points = geoShape.childNodes
        # multi-class
        label = child.getAttribute('name')
        if label != 'airplane':
            continue
        obj_points = []
        for point in points:
            if not (type(point) == xml.dom.minidom.Element):
                continue
            x = float(eval(point.getAttribute("x")))
            y = float(eval(point.getAttribute("y")))
            coords = lonlat2geo(dataset, x, y)
            x = coords[0]
            y = coords[1]
            coords = geo2imagexy(dataset, x, y)
            x = coords[0]
            y = coords[1]
            obj_points.append(abs(int(x)))
            obj_points.append(abs(int(y)))
        objects_points.append(obj_points)
        objects_label.append(label)

    return objects_points, objects_label


def save_voc_xml(save_path, objects, labels, offset_x=0, offset_y=0, width=0, height=0):
    xml_file = open(save_path, 'w')
    xml_file.write('<?xml version="1.0" ?>\n')
    xml_file.write('<annotation>\n')
    xml_file.write('<folder>' + 'JPEGImages' + '</folder>\n')
    xml_file.write('<filename>' + save_path.replace('.xml', '.jpg').split('/')[-1] + '</filename>\n')
    xml_file.write('<path>' + 'xxxx' + '</path>\n')
    xml_file.write('<source>\n')
    xml_file.write('<database>' + 'Unknown' + '</database>\n')
    xml_file.write('<annotation>' + 'xxx' + '</annotation>\n')
    xml_file.write('<image>' + 'xxx' + '</image>\n')
    xml_file.write('<flickrid>' + '0' + '</flickrid>\n')
    xml_file.write('</source>\n')
    xml_file.write('<size>\n')
    xml_file.write('<width>' + str(width) + '</width>\n')
    xml_file.write('<height>' + str(width) + '</height>\n')
    xml_file.write('<depth>3</depth>\n')
    xml_file.write('</size>\n')
    xml_file.write('<segmented>' + '0' + '</segmented>\n')

    for (obj, label) in zip(objects, labels):
        xml_file.write('<object>\n')
        xml_file.write('<name>' + label + '</name>\n')
        xml_file.write('<bndbox>\n')
        xml_file.write('<xmin>' + str(max(1, int(obj[0] - offset_x))) + '</xmin>\n')
        xml_file.write('<ymin>' + str(max(1, int(obj[1] - offset_y))) + '</ymin>\n')
        xml_file.write('<xmax>' + str(min(width - 1, int(obj[2] - offset_x))) + '</xmax>\n')
        xml_file.write('<ymax>' + str(min(height - 1, int(obj[3] - offset_y))) + '</ymax>\n')
        xml_file.write('</bndbox>\n')
        xml_file.write('</object>\n')
    xml_file.write('</annotation>\n')
    xml_file.close()


def img_16bits_to_8bits(src_img, ratio=0.01):
    """
    将图像由16bit转化到8bit
    :param src_img: 待量化拉伸的图像
    :param ratio: 量化拉伸时直方图压缩比例
    :return: 量化拉伸后的图像
    """
    dims = len(src_img.shape)
    if dims == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    elif dims == 2:
        pass
    else:
        print("Error: 16 bit to 8bit, wrong dimension")
        return

    res_img = np.zeros(src_img.shape)
    res_img = np.float32(res_img)
    if src_img.dtype == 'uint16':
        hist = cv2.calcHist([src_img], [0], None, [65536], [0, 65536])
        pixels = src_img.size
        cum_hist = hist.cumsum(0)
        small_cum = ratio*pixels
        high_cum = pixels - small_cum
        smallValue = np.where(cum_hist > small_cum)[0][0]
        highValue = np.where(cum_hist > high_cum)[0][0]
        if highValue == smallValue:
            res_img = np.uint8(res_img)
            return res_img
        src_img = np.where(src_img > highValue, highValue, src_img)
        src_img = np.where(src_img < smallValue, smallValue, src_img)
        scaleRatio = 255.0/(highValue-smallValue)
        src_img = src_img - smallValue
        res_img = src_img * scaleRatio
        res_img = np.uint8(res_img)
    return res_img


def crop_image(objects, labels, img_file, win_size, win_stride, save_dir):

    global total_clips

    dataset = gdal.Open(img_file)
    im_height = dataset.RasterYSize
    im_width = dataset.RasterXSize
    image = dataset.ReadAsArray(0, 0, im_width, im_height) # 整图数据

    boxes = []
    ct = np.zeros((len(objects), 2)) - 1
    for idx, obj in enumerate(objects):
        obj_ = np.int32(obj[:])
        box = obj_.reshape([-1, 2])
        # 多边形标注框->水平矩形框
        box_xmin, box_ymin = np.min(box, axis=0)
        box_xmax, box_ymax = np.max(box, axis=0)

        ct[idx, 0] = np.mean([box_xmin, box_xmax])  # obj中心坐标x_ct
        ct[idx, 1] = np.mean([box_ymin, box_ymax])  # obj中心坐标y_ct

        boxes.append([box_xmin, box_ymin, box_xmax, box_ymax])
    boxes = np.array(boxes)

    for h_start in range(0, image.shape[0], win_stride):
        if h_start <= image.shape[0] - win_size:
            h_start = h_start
        elif h_start > image.shape[0] - win_size and h_start < image.shape[0] - win_size + win_stride:
            h_start = image.shape[0] - win_size
        else:
            break

        for w_start in range(0, image.shape[1], win_stride):
            if w_start <= image.shape[1] - win_size:
                w_start = w_start
            elif w_start > image.shape[1] - win_size and w_start < image.shape[1] - win_size + win_stride:
                w_start = image.shape[1] - win_size
            else:
                break

            local_obj = []
            local_obj_label = []

            xmin = max(0, w_start)
            ymin = max(0, h_start)

            xmax = min(image.shape[1], w_start + win_size)
            ymax = min(image.shape[0], h_start + win_size)
            for idx, obj_ in enumerate(objects):

                if ct[idx][0] > xmin and ct[idx][1] > ymin and ct[idx][0] < xmax and ct[idx][1] < ymax:
                    local_obj.append(boxes[idx])
                    local_obj_label.append(labels[idx])

            if len(local_obj_label) > 0:
                # save annotations
                save_path = os.path.join(save_dir, 'Annotations', '%s_%05d_%05d.xml'
                        % (os.path.basename(img_file).strip('.tiff').split('/')[-1], xmin, ymin))
                if not os.path.exists(os.path.join(save_dir, 'Annotations')):
                    os.makedirs(os.path.join(save_dir, 'Annotations'))
                save_voc_xml(save_path, local_obj, local_obj_label, offset_x=xmin, offset_y=ymin,
                            width=win_size, height=win_size)

                # save tiff or jpg
                # save_path = os.path.join(save_dir, 'JPEGImages', '%s_%05d_%05d.jpg'
                        # % (os.path.basename(img_file).strip('.tiff').split('/')[-1], xmin, ymin))
                save_path = os.path.join(save_dir, 'JPEGImages', '%s_%05d_%05d.tiff'
                                        % (os.path.basename(img_file).strip('.tiff').split('/')[-1], xmin, ymin))

                if not os.path.exists(os.path.join(save_dir, 'JPEGImages')):
                    os.makedirs(os.path.join(save_dir, 'JPEGImages'))
                if len(image.shape) == 2:
                    img_tmp = image[int(ymin):int(ymax), int(xmin):int(xmax)]
                else:
                    img_tmp = image[int(xmin):int(xmax), int(ymin):int(ymax), :]

                # # save jpg
                # img_tmp = img_16bits_to_8bits(img_tmp)
                # cv2.imwrite(save_path, img_tmp)

                # save tiff
                img_tmp = Image.fromarray( img_tmp.astype(np.uint16) )
                img_tmp.save(save_path)

                total_clips += 1

    print("total clips %d" % (total_clips))


def main():

    airbases = os.listdir(DATA_DIR)
    for base in airbases:
        print(base)

        base_dir = os.path.join(DATA_DIR, base)
        for img in os.listdir(base_dir):
            if 'tiff' not in img:
                continue
            if 'aux' in img:
                continue
            print(img)

            img_file = os.path.join(base_dir, img)
            vif_file = img_file.replace('.tiff', '.vif')
            if not os.path.exists(vif_file):
                continue

            objects, labels = read_vif(vif_file, img_file)
            if len(objects) > 0:
                if len(objects[0]) > 0:                    
                    print("**Target Found**")
                    crop_image(objects, labels, img_file, WIN_SIZE, WIN_STRIDE, SAVE_DIR)

            else:
                print("**No Target**")

            print('----------------------------')


if __name__ == '__main__':

    WIN_SIZE = 512
    WIN_STRIDE = 256
    DATA_DIR = '/workspace/RawData/'
    SAVE_DIR = '/workspace/TiffSlice/20221012'
        
    total_clips = 0
    main()
