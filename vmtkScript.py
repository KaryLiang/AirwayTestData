
import os, vtk, time
import pydicom
import numpy as np
# from AirwayModule import *
from skimage.measure import label
from skimage import transform
from scipy import ndimage
from vtkplotter import *
from vmtk import vtkvmtk
import matplotlib
matplotlib.use('qt5agg')
import numba as nb
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import itk

def readImage(path):
    imgNames = os.listdir(path)
    imgs = dict()
    header = dict()
    for name in imgNames:
        dcm = pydicom.read_file(os.path.join(path, name))
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        instanceNumber = int(dcm.InstanceNumber)
        imgs[instanceNumber] = img
        header[instanceNumber] = dcm
    instances = list(imgs.keys())
    instances.sort()
    images = []
    for instance in instances:
        images.append(imgs[instance])
    images = np.array(images, dtype=np.float)
    images = np.transpose(images, axes=[1, 2, 0])
    para = dict()
    para['pixel_spacing'] = float(dcm.PixelSpacing[0])
    assert instances[0]+1 == instances[1]
    pos1 = header[instances[0]].ImagePositionPatient
    pos2 = header[instances[1]].ImagePositionPatient
    spacing_between_slice = np.linalg.norm(np.array(pos1) - np.array(pos2))
    para['spacing_between_slice'] = spacing_between_slice
    return images, para
def generate_target(joints):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''

    target_type = 'gaussian'

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros((len(joints),
                           32, 32, 32),
                          dtype=np.float32)

        sigma = 1
        tmp_size = sigma * 3

        for joint_id in range(len(joints)):
            for point in joints[joint_id]:
                feat_stride = 4
                mu_x = int(point[0] / 4 + 0.5)
                mu_y = int(point[1] / 4 + 0.5)
                mu_z = int(point[2] / 4 + 0.5)
                # # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z + tmp_size + 1)]
                #
                # if ul[0] >= 32 or ul[1] >= 32 or ul \
                #         or br[0] < 0 or br[1] < 0:
                #     # If not, just return the image as is
                #     continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                z = y[::, np.newaxis]
                x0 = y0 = z0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0)**2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], 32) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], 32) - ul[1]
                g_z = max(0, -ul[2]), min(br[2], 32) - ul[2]
                # Image range
                img_x = max(0, ul[0]), min(br[0], 32)
                img_y = max(0, ul[1]), min(br[1], 32)
                img_z = max(0, ul[2]), min(br[2], 32)

                # target[joint_id][img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                #     g[g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]]

                patch1 = target[joint_id][img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]]
                patch2 = g[g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]]
                idx = patch2 > patch1
                patch1[idx] = patch2[idx]
                target[joint_id][img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = patch1

    return target
def getBifurcation(lines):
    '''
    To get bifurcations
    param lines: list, each element of it is a line, and has the shape of (m, 3), where m is the number of the points of the line
    return bifurcations: numpy array, (n, 3), n is the number of the bifurcations
    '''
    matShape = [1000, 1000, 1000]
    degreeMat = np.zeros(matShape)
    for line in lines:
        num = line.shape[0] - 1
        x1, y1, z1 = int(line[0, 0]), int(line[0, 1]), int(line[0, 2])
        x2, y2, z2 = int(line[num, 0]), int(line[num, 1]), int(line[num, 2])

        degreeMat[x1,y1,z1] += 1
        degreeMat[x2, y2, z2] += 1

    bifurcations = np.where(degreeMat>1)
    bifurcations = np.transpose(np.array(bifurcations), (1,0))
    return bifurcations

def exchangePoint(line):
    tempPoint = line["endpoint1"]
    line["endpoint1"] = line["endpoint2"]
    line["endpoint2"] = tempPoint
    temFlag = line["p1IsBif"]
    line["p1IsBif"] = line["p2IsBif"]
    line["p2IsBif"] = temFlag

def sortEndpoints(lines):
    '''
    to make line of "endpoint1" as start point and "endpoint2" as end point
    '''
    sortedLines = []
    # to find the start line
    for i, line in enumerate(lines):
        tmpLines = []
        if line["order"] == 1:
            if line["p1IsBif"] == 1 and line["p2IsBif"] == 0:
                exchangePoint(line) #exchange endpoints
            tmpLines.append(line)
            sortedLines.append(line)
            lines.pop(i)
            break

    while(lines):
        curLines = []
        order = tmpLines[0]["order"]+1
        for i in range(len(lines)-1, -1, -1):
            if lines[i]["order"] == order:

                for tmpL in tmpLines:
                    if (lines[i]["endpoint2"] == tmpL["endpoint2"]).all() or (lines[i]["endpoint2"] == tmpL["endpoint1"]).all():
                        exchangePoint(lines[i]) #exchange endpoints
                        break
                sortedLines.append(lines[i])
                curLines.append(lines.pop(i))
        tmpLines = curLines.copy()
    return sortedLines

def arrayInList(dataArr, dataList):
    flag = 0
    for data in dataList:
        if dataArr[0] == data[0] and dataArr[1] == data[1] and dataArr[2] == data[2]:
            flag = 1
    return flag

@nb.jit()
def getIds(surface, point):

    points = surface.GetPoints()
    n = points.GetNumberOfPoints()
    minDistance = 1E10
    id = -1
    for i in range(n):
        p = points.GetPoint(i)
        dis = np.linalg.norm(np.array(p) - np.array(point))
        if dis < minDistance:
            minDistance = dis
            id = i
    return id

if __name__ == '__main__':

    d = np.load('.\keypoints\data\PA000019.npz', allow_pickle=True)
    mask = d['isoMask']
    # mask = ndimage.interpolation.zoom(mask, zoom=2, order=0)
    # dilate
    maskITK = itk.GetImageFromArray(mask)
    PixelType = itk.UC
    Dimension = 3
    radiusValue = 3 # 8
    foregroundValue = np.max(mask)
    ImageType = itk.Image[PixelType, Dimension]
    StructuringElementType = itk.FlatStructuringElement[Dimension]
    structuringElement = StructuringElementType.Ball(radiusValue)
    DilateFilterType = itk.BinaryDilateImageFilter[ImageType,
                                                   ImageType,
                                                   StructuringElementType]
    dilateFilter = DilateFilterType.New()
    dilateFilter.SetInput(maskITK)
    dilateFilter.SetKernel(structuringElement)
    dilateFilter.SetForegroundValue(1) # Value to dilate

    lines = d['skeleton']
    source = np.flip(np.array(lines[0]['endpoint1']),0)

    # img = Volume(mask)
    img = Volume(itk.GetArrayFromImage(dilateFilter.GetOutput()))
    surface = img.isosurface(1)
    # show(surface)
    surface = surface._polydata

    surfaceCleaner = vtk.vtkCleanPolyData()
    surfaceCleaner.SetInputData(surface)
    surfaceCleaner.Update()

    surfaceTriangulator = vtk.vtkTriangleFilter()
    surfaceTriangulator.SetInputConnection(surfaceCleaner.GetOutputPort())
    surfaceTriangulator.PassLinesOff()
    surfaceTriangulator.PassVertsOff()
    surfaceTriangulator.Update()

    centerlineInputSurface = surfaceTriangulator.GetOutput()

    surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
    surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
    surfaceCapper.SetDisplacement(0.0)
    surfaceCapper.SetInPlaneDisplacement(0.0)
    surfaceCapper.Update()
    centerlineInputSurface = surfaceCapper.GetOutput()
    capCenterIds = surfaceCapper.GetCapCenterIds()

    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(centerlineInputSurface)
    pointLocator.BuildLocator()


    sourceId = vtk.vtkIdList()
    idsource = pointLocator.FindClosestPoint(source)
    sourceId.InsertNextId(idsource)

    targetId = vtk.vtkIdList()
    allTargetId = []
    allSourceId = []
    for idx, line in enumerate(lines):

        if line['p2IsBif'] == 0:
            idtarget = pointLocator.FindClosestPoint(np.flip(np.array(line['endpoint2']), 0))
            allTargetId.append(idtarget)
            targetId.InsertNextId(idtarget)
            print(lines[0]['endpoint1'], idsource, line['endpoint2'], idtarget)


    start = time.time()
    centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
    centerlineFilter.SetInputData(centerlineInputSurface)
    centerlineFilter.SetSourceSeedIds(sourceId)
    centerlineFilter.SetTargetSeedIds(targetId)
    centerlineFilter.SetRadiusArrayName('MaximumInscribedSphereRadius')
    centerlineFilter.SetCostFunction('1/R')
    centerlineFilter.SetFlipNormals(0)
    centerlineFilter.SetAppendEndPointsToCenterlines(0)
    centerlineFilter.SetSimplifyVoronoi(0)
    centerlineFilter.SetCenterlineResampling(0)
    centerlineFilter.SetResamplingStepLength(1.0)
    centerlineFilter.Update()
    end = time.time()
    print(end-start)
    Centerlines = centerlineFilter.GetOutput()
    centerLinePoints = Centerlines.GetPoints()
    voronoiDiagram = centerlineFilter.GetVoronoiDiagram()

    voronoiPoints = vtk_to_numpy(voronoiDiagram.GetPoints().GetData())
    voronoiRadius = vtk_to_numpy(voronoiDiagram.GetPointData().GetArray('MaximumInscribedSphereRadius'))

    lines = Centerlines.GetLines()
    numberOfLines = lines.GetNumberOfCells()

    points = Centerlines.GetPoints()
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    linesList = []
    while lines.GetNextCell(idList):
        n = idList.GetNumberOfIds()
        lineList = []
        for id in range(n):
            # print(idList.GetId(id))
            pointId = idList.GetId(id)
            point = points.GetPoint(pointId)
            lineList.append(list(point))
        linesList.append(lineList)

    vtkLines = []
    for lineList in linesList:
        if len(lineList) == 2:
            continue
        startPoints = []
        for idx, point in enumerate(lineList):
            if idx == 0:
                continue
            startPoints.append([lineList[idx-1], lineList[idx]])
        vtkLine = Lines(startPoints, c='red')
        vtkLines.append(vtkLine)
    # show(img, vtkLines)
    # mask = transform.resize(mask, [mask.shape[0]*2, mask.shape[1]*2, mask.shape[2]*2])
    img = Volume(mask)
    # img = Volume(itk.GetArrayFromImage(dilateFilter.GetOutput()))
    surface = img.isosurface(1)
    # show(surface)
    surface._polydata = centerlineInputSurface
    show(surface.alpha(0.2).color('blue'), vtkLines)
    #
