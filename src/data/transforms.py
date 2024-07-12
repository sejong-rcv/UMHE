import cv2
import numpy as np
# import porespy as ps

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography
from PIL import Image
import torch
import torchvision.transforms as T

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, bigger of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, random_seed=None):
        assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                src_ratio = h / w
                target_w, target_h = self.output_size
                if src_ratio < target_h / target_w:
                    new_w, new_h = (int(np.round(target_h / src_ratio)), target_h)
                else:
                    new_w, new_h = (target_w, int(np.round(target_w * src_ratio)))

            new_h, new_w = int(new_h), int(new_w)
            images[i] = cv2.resize(images[i], (new_w, new_h))

        return images, targets


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            new_h, new_w = self.output_size

            if h != new_h:
                top = np.random.randint(0, h - new_h)
            else:
                top = 0
            if w != new_w:
                left = np.random.randint(0, w - new_w)
            else:
                left = 0

            images[i] = images[i][top: top + new_h, left: left + new_w]

        return images, targets


class CenterCrop(object):
    """Crop center the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random_seed=None):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            new_w, new_h = self.output_size

            if h != new_h:
                top = (h - new_h)//2
            else:
                top = 0
            if w != new_w:
                left = (w - new_w)//2
            else:
                left = 0

            images[i] = images[i][top: top + new_h, left: left + new_w]

        return images, targets


class ImageConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class ImageConvertToInts(object):
    def __call__(self, image):
        return np.rint(image).astype(np.uint8)


class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ImageRandomBrightness(object):
    def __init__(self, max_delta=32, random_state=None):
        assert max_delta >= 0.0
        assert max_delta <= 255.0
        self.delta = max_delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            delta = self.random_state.uniform(-self.delta, self.delta)
            image += delta
        return image


class ImageRandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    # expects float image
    def __call__(self, image):
        if self.random_state.randint(2):
            alpha = self.random_state.uniform(self.lower, self.upper)
            image *= alpha
        return image


class ImageConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image


class ImageRandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 1] *= self.random_state.uniform(self.lower, self.upper)
        return image


class ImageRandomHue(object):
    def __init__(self, delta=18.0, random_state=None):
        assert 0.0 <= delta <= 360.0
        self.delta = delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 0] += self.random_state.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class ImageSwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class ImageRandomLightingNoise(object):
    def __init__(self, random_state):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            swap = self.perms[self.random_state.randint(len(self.perms))]
            shuffle = ImageSwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class PhotometricDistort(object):
    def __init__(self, keys, random_state=None):
        self.random_state = random_state
        self.pd = [
            ImageRandomContrast(random_state=self.random_state),  # RGB
            ImageConvertColor(current="RGB", transform='HSV'),  # HSV
            ImageRandomSaturation(random_state=self.random_state),  # HSV
            ImageRandomHue(random_state=self.random_state),  # HSV
            ImageConvertColor(current='HSV', transform='RGB'),  # RGB
            ImageRandomContrast(random_state=self.random_state)  # RGB
        ]
        self.from_int = ImageConvertFromInts()
        self.rand_brightness = ImageRandomBrightness(random_state=self.random_state)
        self.rand_light_noise = ImageRandomLightingNoise(random_state=self.random_state)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            im = data[key].copy()
            im = self.from_int(im)
            im = self.rand_brightness(im)
            if self.random_state.randint(2):
                distort = ImageCompose(self.pd[:-1])
            else:
                distort = ImageCompose(self.pd[1:])
            im = distort(im)
            im = self.rand_light_noise(im)
            data[key] = im
        return data


class PhotometricDistortSimple(object):
    def __init__(self, keys, max_delta=32, random_state=None):
        self.random_state = random_state
        self.max_delta = max_delta

        lower = 1.0 - self.max_delta / 32 * 0.5
        upper = 1.0 + self.max_delta / 32 * 0.5
        self.pd = [
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state),  # RGB
            ImageConvertColor(current="RGB", transform='HSV'),  # HSV
            ImageRandomSaturation(lower=lower, upper=upper*1.5, random_state=self.random_state),  # HSV
            ImageRandomHue(delta=max_delta/2, random_state=self.random_state),  # HSV
            ImageConvertColor(current='HSV', transform='RGB'),  # RGB
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state)  # RGB
        ]
        self.from_int = ImageConvertFromInts()
        self.rand_brightness = ImageRandomBrightness(max_delta=max_delta, random_state=self.random_state)
        if max_delta > 0:
            self.rand_light_noise = ImageRandomLightingNoise(random_state=self.random_state)
        self.keys = keys

    def __call__(self, data, isTher=False):
        for key in self.keys:
            im = data[key].copy()
            im = self.from_int(im)
            im = self.rand_brightness(im)

            if not isTher:
                if self.random_state.randint(2):
                    distort = ImageCompose(self.pd[:-1])
                else:
                    distort = ImageCompose(self.pd[1:])
            else:
                if self.random_state.randint(2):
                    distort = ImageCompose(self.pd[0:1])
                else:
                    distort = None

            if distort is not None:
                im = distort(im)

            data[key] = im
        return data


class ToGrayscale(object):
    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            # RGB 2 GRAY
            images[i] = np.expand_dims(images[i][:, :, 0] * 0.299 +
                                       images[i][:, :, 1] * 0.587 +
                                       images[i][:, :, 2] * 0.114, axis=-1)
        return images, targets

class DictToRGB(object):
    def __init__(self, keys, *args):
        self.keys = keys

    def __call__(self, data):

        for key in self.keys:
            
            data[key] = data[key] / 255
                
        return data

class DictToGrayscale(object):
    def __init__(self, keys, *args):
        self.keys = keys

    def __call__(self, data):

        for key in self.keys:
            
            if data['pairs_flag'] == 0 : #RGB
                data[key] = np.expand_dims(data[key][:, :, 0] * 0.299 +
                                           data[key][:, :, 1] * 0.587 +
                                           data[key][:, :, 2] * 0.114, axis=-1)

            elif data['pairs_flag'] == 1 :#Thermal
                data[key] = np.expand_dims(data[key][:, :, 0], axis=-1)

            elif data['pairs_flag'] == 2 :
                if key == 'patch_1' or key =='image_1' :
                    # RGB 2 GRAY
                    data[key] = np.expand_dims(data[key][:, :, 0] * 0.299 +
                                               data[key][:, :, 1] * 0.587 +
                                               data[key][:, :, 2] * 0.114, axis=-1)
                else : 
                    data[key] = np.expand_dims(data[key][:, :, 0], axis=-1)

            else :
                if key == 'patch_2' or key =='image_2' :
                    # RGB 2 GRAY
                    data[key] = np.expand_dims(data[key][:, :, 0] * 0.299 +
                                               data[key][:, :, 1] * 0.587 +
                                               data[key][:, :, 2] * 0.114, axis=-1)
                else : 
                    data[key] = np.expand_dims(data[key][:, :, 0], axis=-1)

            data[key] = data[key] / 255
                
        return data


class Standardize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            images[i] = (images[i].astype(np.float32)/255 - self.mean) / self.std
        return images, targets


class DictStandardize(object):
    def __init__(self, mean, std, keys, *args):
        self.mean = mean
        self.std = std
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = (data[key].astype(np.float32)/255 - self.mean) / self.std
        return data

class Multispectral_DictStandardize(object):
    def __init__(self, mean, std, keys, *args):
        self.mean = mean[0]
        self.mean_lwir = mean[1]
        self.std = std[0]
        self.std_lwir = std[1]
        self.keys = keys

    def __call__(self, data):

        for key in self.keys:

            if data['pairs_flag'] == 0 :
                if key == 'patch_2' or key=='image_2' :
                    data[key] = (data[key].astype(np.float32)/255 - self.mean_lwir) / self.std_lwir
                else :
                    data[key] = (data[key].astype(np.float32)/255 - self.mean) / self.std
            elif data['pairs_flag'] == 1 : 
                data[key] = (data[key].astype(np.float32)/255 - self.mean) / self.std
            else :
                data[key] = (data[key].astype(np.float32)/255 - self.mean_lwir) / self.std_lwir

        return data
    
class ToTensorWithTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            images[i] = images[i].transpose((2, 0, 1))

        # Targets
        if targets is not None:
            targets = torch.from_numpy(np.array(targets))

        return torch.from_numpy(np.array(images)), targets


class ChangeAwarePrep(object):
    """
    @TODO: Describe it!
    """

    def __init__(self, keys=['image', 'positive', 'weak_positive']):
        self.keys = keys

    def __call__(self, data):

        images, targets = data
        assert len(images) == len(self.keys), 'Something is weid: len(images)={}  len(self.keys)=={}'.format(
            len(images), len(self.keys)
        )

        ret_dict = {}
        for i, k in enumerate(self.keys):
            ret_dict[k] = images[i]

        return ret_dict



class Multispectral_HomographyNetPrep(object):
    """
    Data preparation procedure like in the [1].
    "To  generate  a  single  training  example,  we  first  randomly crop a square patch from the larger image
     I at position p (we avoid  the  borders  to  prevent  bordering  artifacts  later  in  the data  generation
     pipeline). This  random  crop  is Ip.  Then,  the four  corners  of  Patch  A  are  randomly  perturbed  by
     values within  the  range  [-ρ,ρ]. The  four  correspondences  define a  homography HAB.  Then, the  inverse
     of  this  homography HBA= (HAB)−1 is  applied  to the  large  image  to  produce image I′. A second patch I′p
     is cropped from I′ at position p. The two grayscale patches, Ip and I'p are then stacked channelwise to create
     2-channel image which is fed directly to our ConvNet. The 4-point parametrization of HAB is then used as the
     associated ground-truth training label."
    [1] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep Image Homography Estimation. ArXiv, abs/1606.03798.
    Args:
        rho (int): point perturbation range.
        patch_size (int): size of patch.
    """

    def __init__(self, rho, patch_size, photometric_distort_keys=None, max_delta=32, target_gen='4_points', trainsplit=None, trainprop=None, random_seed=42):
        self.rho = rho
        self.patch_size = patch_size
        self.target_gen = target_gen
        self.photometric_distort_keys = photometric_distort_keys
        self.max_delta = max_delta
        self.random_seed = random_seed
        self.rgb_jitter = T.ColorJitter( brightness=.2, contrast=.2)
        self.rgb_transforms = T.Compose([self.rgb_jitter])
        self.toGray = T.Grayscale()
        self.ther_jitter = T.ColorJitter(brightness=.2, contrast=.2)
        if trainsplit == 'None' :
            self.trainsplit = None
        else :
            self.trainsplit = trainsplit
        
        if trainprop == 'None' : 
            self.trainprop = None
        else :
            self.trainprop = trainprop
        
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)
            self.randint_fn = self.random_state.randint
        else:
            self.random_state = np.random
            self.randint_fn = np.random.randint
    def save_im(self, img, name):
        import os
        cv2.imwrite(os.path.join("debug", name), cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    def window_partition(self, x, window_size):
        H, W, C = x.shape

        x = x.reshape(window_size, H // window_size, window_size, W // window_size, C)

        windows = np.ascontiguousarray(x.transpose(0,2,1,3,4))

        return windows

    def partition_merge(self, x):
        window_size, _, H, W, C = x.shape

        windows = np.ascontiguousarray(x.transpose(0,2,1,3,4).reshape(H*window_size, W*window_size, C))

        return windows

    def set_rand_range(self, num, i):
        if i%2==0:
            row = np.concatenate([np.random.randint(-num, num // 4, size=(1)), np.random.randint(-num, num, size=(1)), np.random.randint(-num // 4, num, size=(1))])
            col = np.concatenate([np.random.randint(-num // 4, num, size=(1)), np.random.randint(-num, num, size=(1)), np.random.randint(-num, num // 4, size=(1))])
        else:
            col = np.concatenate([np.random.randint(-num, num // 4, size=(1)), np.random.randint(-num, num, size=(1)), np.random.randint(-num // 4, num, size=(1))])
            row = np.concatenate([np.random.randint(-num // 4, num, size=(1)), np.random.randint(-num, num, size=(1)), np.random.randint(-num, num // 4, size=(1))])

        return row, col

    def __call__(self, data):

        images, targets = data

        # Get image
        image = images[0][0] #Real Thermal
        rgb = images[0][1] #Real RGB
        
        h, w = image.shape[:2]
        rho = self.rho

        #################################################original##############################################
        if self.trainsplit != None :
            if np.random.rand(1) >= self.trainprop[0]:

                if np.random.rand(1) >= self.trainprop[1]:
                    pairs_flag = 'rgb'
                    image_1 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
                    
                    if images[1][1].shape[-1] == 3:
                        image_2 = images[1][1][:,:,0:1]
                    else:
                        image_2 = images[1][1]

                    if 'image_2' in self.photometric_distort_keys:
                        
                        image_2 = np.asarray(self.rgb_transforms(Image.fromarray(np.repeat(image_2,3,axis=-1))))

                        image_2 = image_2[:,:,0:1]

                        
                else :
                    pairs_flag = 'thermal'
                    image_1 = np.copy(image)[:,:,0:1]

                    if 'image_2' in self.photometric_distort_keys:
                        image_2 = images[1][0]
                        if image_2.shape[-1]==3:
                            image_2 = image_2[:,:,0:1]

                        image_2 = np.asarray(self.ther_jitter(Image.fromarray(np.repeat(image_2, 3, axis=-1))))[:,:,0:1]

            else :
                if np.random.rand(1) >= self.trainprop[1]:
                    pairs_flag = 'multispectral'
                    image_1 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
                    image_2 = image[:,:,0:1]
                else :

                    if np.random.rand(1) >= self.trainprop[1]:
                        pairs_flag = 'multispectral_homography'
                        image_1 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
                        image_2 = image[:,:,0:1]
                    else:
                        pairs_flag = 'multispectral_inv_homography'
                        image_1 = image[:,:,0:1]
                        image_2 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
                
        else :
            pairs_flag = 'multispectral'
            image_1 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
            image_2 = image[:,:,0:1]

        if self.patch_size != w:
            pos_x = self.randint_fn(self.rho + self.patch_size // 2, w - self.rho - self.patch_size // 2 + 1)#(252~389) randomint
            pos_y = self.randint_fn(self.rho + self.patch_size // 2, h - self.rho - self.patch_size // 2 + 1)#(252~261) randomint

        else:
            pos_x = w//2
            pos_y = h//2

        # Get patch coords (x/y)
        corners = np.array([(pos_x - self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y + self.patch_size // 2),
                            (pos_x - self.patch_size // 2, pos_y + self.patch_size // 2)]) #left top, right top, right bot, left bot
        
        patch_1 = image_1[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
        patch_2 = image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
        
        delta = self.randint_fn(-rho, rho, 8).reshape(4, 2)
        
        #######################################################################
        # Get perspective transform NEW
        #######################################################################

        # Calc homography between
        homography = four_point_to_homography(np.expand_dims(corners, axis=0), np.expand_dims(delta, axis=0),crop=False)

        if self.trainsplit != None and pairs_flag != 'multispectral':
            if pairs_flag == 'rgb':
                nopd_image_2 = np.expand_dims(np.asarray(self.toGray(Image.fromarray(np.copy(rgb)))), axis=-1)
            elif pairs_flag == 'thermal':
                nopd_image_2 = np.copy(image)[:,:,0:1]
            else:
                nopd_image_2 = np.copy(image_2)

            image_2 = warp_image(image_2, homography, target_h=image_2.shape[0], target_w=image_2.shape[1])
            nopd_image_2 = warp_image(np.copy(nopd_image_2), homography, target_h=image_2.shape[0], target_w=image_2.shape[1])

            if len(image_2.shape) == 2:
                image_2 = np.expand_dims(image_2, axis=-1)
                nopd_image_2 = np.expand_dims(nopd_image_2, axis=-1)

            patch_2 = image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
            nopd_patch_2 = nopd_image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]

            if len(patch_2.shape) == 2:
                patch_2 = np.expand_dims(patch_2, axis=-1)
                nopd_patch_2 = np.expand_dims(nopd_patch_2, axis=-1)
        else:
            nopd_patch_2 = patch_2
        ###################################################################
        # Prepare output data - 4 points
        ###################################################################

        if self.target_gen == '4_points':
            target = delta

        else:
            assert False, 'I do not know this, it should be either \'4_points\' ar \'all_points\''

        
        if pairs_flag == 'rgb' : 
            pairs_flag = 0
        elif pairs_flag == 'thermal' : 
            pairs_flag = 1 
        elif pairs_flag == 'multispectral' or pairs_flag == 'multispectral_homography': 
            pairs_flag = 2
        else :
            pairs_flag = 3
        
        return {'image_1': 1, 'image_2': 2, 'patch_1': patch_1, 'patch_2': patch_2, 'corners': corners,
                'target': target, 'delta': delta, 'homography': homography, 'pairs_flag': pairs_flag,'nopd_patch_2': nopd_patch_2}
    
class Multispectral_HomographyNetPrep_test(object):
    """
    Data preparation procedure like in the [1].
    "To  generate  a  single  training  example,  we  first  randomly crop a square patch from the larger image
     I at position p (we avoid  the  borders  to  prevent  bordering  artifacts  later  in  the data  generation
     pipeline). This  random  crop  is Ip.  Then,  the four  corners  of  Patch  A  are  randomly  perturbed  by
     values within  the  range  [-ρ,ρ]. The  four  correspondences  define a  homography HAB.  Then, the  inverse
     of  this  homography HBA= (HAB)−1 is  applied  to the  large  image  to  produce image I′. A second patch I′p
     is cropped from I′ at position p. The two grayscale patches, Ip and I'p are then stacked channelwise to create
     2-channel image which is fed directly to our ConvNet. The 4-point parametrization of HAB is then used as the
     associated ground-truth training label."
    [1] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep Image Homography Estimation. ArXiv, abs/1606.03798.
    Args:
        rho (int): point perturbation range.
        patch_size (int): size of patch.
    """

    def __init__(self, rho, patch_size, photometric_distort_keys=None, max_delta=32, target_gen='4_points', trainsplit=None, trainprop=None, random_seed=42):
        self.rho = rho
        self.patch_size = patch_size
        self.target_gen = target_gen
        self.photometric_distort_keys = photometric_distort_keys
        self.max_delta = max_delta
        self.random_seed = random_seed
        self.toGray = T.Grayscale()
        if trainsplit == 'None' :
            self.trainsplit = None
        else :
            self.trainsplit = trainsplit
        
        if trainprop == 'None' : 
            self.trainprop = None
        else :
            self.trainprop = trainprop
        
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)
            self.randint_fn = self.random_state.randint
        else:
            self.random_state = np.random
            self.randint_fn = np.random.randint

    def __call__(self, data):

        images, targets = data

        # Get image
        image = images[0][0][:,:,0:1]
        rgb = np.expand_dims(np.asarray(self.toGray(Image.fromarray(images[0][1]))), axis=-1)
        
        h, w = image.shape[:2]
        
        
        if self.trainsplit != None :
            if np.random.rand(1) >= self.trainprop[0]:
                if np.random.rand(1) >= self.trainprop[1]:
                    pairs_flag = 'rgb'
                    image_1 = np.copy(rgb)
                    image_2 = np.copy(rgb)
                    if 'image_1' in self.photometric_distort_keys:
                        image_1 = PhotometricDistortSimple(keys=['image_1'], max_delta=self.max_delta,
                                               random_state=self.random_state)({'image_1': image_1})['image_1']
                    if 'image_2' in self.photometric_distort_keys:
                        image_2 = PhotometricDistortSimple(keys=['image_2'], max_delta=self.max_delta,
                                                           random_state=self.random_state)({'image_2': image_2})['image_2']

                else :
                    pairs_flag = 'thermal'
                    image_1 = np.copy(image)
                    image_2 = np.copy(image)
                    if 'image_1' in self.photometric_distort_keys:
                        image_1 = PhotometricDistortSimple(keys=['image_1'], max_delta=self.max_delta,
                                               random_state=self.random_state)({'image_1': image_1})['image_1']
                    if 'image_2' in self.photometric_distort_keys:
                        image_2 = PhotometricDistortSimple(keys=['image_2'], max_delta=self.max_delta,
                                                           random_state=self.random_state)({'image_2': image_2})['image_2']
            else :
                pairs_flag = 'multispectral'
                image_1 = rgb
                image_2 = image
        else :
            pairs_flag = 'multispectral'
            image_1 = rgb
            image_2 = image
            
        # Calc position of patch center
        if self.patch_size != w:
            pos_x = self.randint_fn(self.rho + self.patch_size // 2, w - self.rho - self.patch_size // 2 + 1)
            pos_y = self.randint_fn(self.rho + self.patch_size // 2, h - self.rho - self.patch_size // 2 + 1)
        else:
            pos_x = w//2
            pos_y = h//2


        # Get patch coords (x/y)
        corners = np.array([(pos_x - self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y + self.patch_size // 2),
                            (pos_x - self.patch_size // 2, pos_y + self.patch_size // 2)])
        
        patch_1 = image_1[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
        patch_2 = image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
        
        delta = self.randint_fn(-self.rho, self.rho, 8).reshape(4, 2)
        
        #######################################################################
        # Get perspective transform NEW
        #######################################################################

        # Calc homography between
        homography = four_point_to_homography(np.expand_dims(corners, axis=0), np.expand_dims(delta, axis=0),crop=False)
        
        if self.trainsplit != None and pairs_flag != 'multispectral':
            image_2 = warp_image(image_2, homography, target_h=image_2.shape[0], target_w=image_2.shape[1])
            if len(image_2.shape) == 2:
                image_2 = np.expand_dims(image_2, axis=-1)
            patch_2 = image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
            if len(patch_2.shape) == 2:
                patch_2 = np.expand_dims(patch_2, axis=-1)

        ###################################################################
        # Prepare output data - 4 points
        ###################################################################

        if self.target_gen == '4_points':
            target = delta

        ###################################################################
        # Prepare output data - all points
        ###################################################################

        elif self.target_gen == 'all_points':

            # Create grid of targets
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            point_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            # Transform grid o points
            point_grid_t = cv2.perspectiveTransform(np.array([point_grid], dtype=np.float32), homography).squeeze()
            diff_grid_t = point_grid_t - point_grid
            diff_x_grid_t = diff_grid_t[:, 0]
            diff_y_grid_t = diff_grid_t[:, 1]
            diff_x_grid_t = diff_x_grid_t.reshape((h, w))
            diff_y_grid_t = diff_y_grid_t.reshape((h, w))

            pf_patch_x_branch = diff_x_grid_t[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
            pf_patch_y_branch = diff_y_grid_t[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]

            target = np.zeros((self.patch_size, self.patch_size, 2))
            target[:, :, 0] = pf_patch_x_branch
            target[:, :, 1] = pf_patch_y_branch

        else:
            assert False, 'I do not know this, it should be either \'4_points\' ar \'all_points\''

        
        if pairs_flag == 'rgb' : 
            pairs_flag = 0
        elif pairs_flag == 'thermal' : 
            pairs_flag = 1 
        elif pairs_flag == 'multispectral' : 
            pairs_flag = 2
        
        return {'image_1': image_1, 'image_2': image_2, 'patch_1': patch_1, 'patch_2': patch_2, 'corners': corners,
                'target': target, 'delta': delta, 'homography': homography, 'pairs_flag': pairs_flag, 'nopd_patch_2': 1}

class Key_change(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, keys=['patch','image'], *args):
        self.keys = keys

    def __call__(self, data):
        
        tmp = data['image_1']
        data['image_1'] = data['patch_1']
        data['patch_1'] = tmp
        
        tmp = data['image_2']
        data['image_2'] = data['patch_2']
        data['patch_2'] = tmp
        
        return data
    
class DictToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, keys=['image', 'positive', 'weak_positive'], *args):
        self.keys = keys

    def __call__(self, data):
        for key in data:
            if key in self.keys:
                if len(data[key].shape) == 3:
                    # swap color axis because
                    # numpy image: H x W x C
                    # torch image: C X H X W
                    data[key] = data[key].transpose((2, 0, 1))
            data[key] = torch.from_numpy(np.array(data[key]))
        return data


class CollatorWithBlobs(object):

    def __init__(self, patch_1_key=None, patch_2_key=None, blob_porosity=None, blobiness=None, random_seed=None):
        self.patch_1_key = patch_1_key
        self.patch_2_key = patch_2_key
        self.blob_porosity = blob_porosity
        self.blobiness = blobiness
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)
            self.rand_choice_fn = self.random_state.choice
        else:
            self.rand_choice_fn = np.random.choice

    def __call__(self, batch):

        ###################################################################
        # Collate
        ###################################################################

        keys = list(batch[0].keys())
        output_dict = {key: [] for key in keys}
        for elem in batch:
            for key in keys:
                output_dict[key].append(elem[key])
        for key in keys:
            output_dict[key] = torch.stack(output_dict[key])

        ###################################################################
        # Generate blobs
        ###################################################################
        if self.patch_1_key is not None:

            h, w = output_dict[self.patch_1_key].shape[-2:]
            for elem_idx in range(len(batch)):
                # Pick image to copy content from
                possible_indices = np.arange(len(batch))
                possible_indices = np.delete(possible_indices, np.where(possible_indices == elem_idx))
                other_index = self.rand_choice_fn(possible_indices, 1)[0]

                # Create blob
                blobs = ps.generators.blobs(shape=[h, w], porosity=self.blob_porosity, blobiness=self.blobiness)
                blobs = torch.from_numpy(blobs)

                # Copy
                patch_1 = output_dict[self.patch_1_key][other_index]
                patch_2 = output_dict[self.patch_2_key][elem_idx]
                blobs = blobs.unsqueeze(0).repeat((patch_2.shape[0], 1, 1))
                patch_2_new = torch.mul(patch_1, blobs)
                patch_2_old = torch.mul(patch_2, torch.bitwise_not(blobs))
                patch_2_augmented = patch_2_old + patch_2_new
                output_dict[self.patch_2_key][elem_idx] = patch_2_augmented

        return 