import imgaug as ia
from imgaug import augmenters as iaa

# TODO
# input : (image, [truthbox])
# output : augmented (image, [truthbox])

param

seq = iaa.Squential(
        [
            iaa.OneOf([
                iaa.Noop(),
                iaa.Crop(percent=(0.0,0.9),keep_size=False),
                iaa.OneOf([
                    iaa.Crop()
                    ])
                ])
            iaa.
            ]
        )
