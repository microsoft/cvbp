# Computer Vision Best Practice

Giving computers some semblance of vision (and the implied ability to
process the images and identify the content) has become practical with
the massive compute and training data that is available today. A
number of pre-built machine learning models are freely available for
this task. 

This [MLHub](https://mlhub.ai) package provides a demonstration of
Microsoft's open source Computer Vision Best Practice repository
available from <https://github.com/microsoft/ComputerVision>.  The
package supplies an interactive demonstration as an overview of the
capabilities of the repository. Individual command line tools are also
packaged for common computer vision tasks based on a collection of
pre-built computer vision models. These command line tools aim to be
used as part of a Linux pipeline of image processing commands.

Visit the github repository for this package for more details:
<https://github.com/microsoft/cvbp>

## Quick Start Command Line Examples

```console
$ ml demo cvbp
$ ml classify cvbp
$ ml classify cvbp https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
```

## Usage

- To install mlhub (Ubuntu)

```console
$ pip3 install mlhub
```

- To install and configure the package:

```console
$ ml install   cvbp
$ ml configure cvbp
```

## Command Line Tools

In addition to the *demo* presented below, the *cvbp* package provides
useful command line tools. Below we demonstrate a number of
these. Most commands take an image as a parameter which may be a url
or a path to a local file.

**classify**

The *classify* command will identify the dominant object in a photo
with a level of confidence. The confidence, class, and filename are
printed and can be piped on to other commands within a command
line. This can allow us to add the class as a tag to the meta data of
an image file.

This first example image from the Internet is classified 100% as a
koala.

![](https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG)
```console
$ ml classify cvbp https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
1.00,koala,https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
```

As an example of command line processing, we could download this photo
(noting its license) and add the appropriate tag to it:

```console
$ wget -O koala.jpg https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
$ exiftool koala.jpg | grep -i comment
$ ml classify cvbp koala.jpg |
  cut -d, -f2 |
  xargs bash -c 'mogrify -comment $0 koala.jpg' 
$ exiftool koala.jpg | grep -i comment
Comment                         : koala
```

Coffee mugs seem to be fairly standard fare for image classification:

![](https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg)
```console
$ ml classify cvbp https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg
0.68,coffee_mug,https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg
```

Different pre-built image classification models are available. Here we
use a more complex model to provide a more confident classification of
the coffee mug:

```console
$ ml classify cvbp --model=resnet152 https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg
0.85,coffee_mug,https://cvbp.blob.core.windows.net/public/images/cvbp_cup.jpg
```

If no image is supplied on the command line then the computer's webcam
will be used to source a live feed and any objects held up to the
camera will be classified. The live classification is available within
the displayed live video image.

```console
$ ml classify cvbp
```

Multiple images can be classified with one command line:

```console
$ ml classify cvbp images/*.png
0.34,jay,images/image_01_bw.png
0.99,custard_apple,images/image_02_bw.png
0.96,echidna,images/image_03_bw.png
0.65,Afghan_hound,images/image_04_bw.png
0.99,beacon,images/image_05_bw.png
0.33,Tibetan_terrier,images/image_06_bw.png
0.96,great_white_shark,images/image_07_bw.png
0.58,crayfish,images/image_09_bw.png
0.77,redshank,images/image_10_bw.png
```

We can add a tag to photos which are classified with a confidence
greater than 75%. This might allow us to later on search for photos
using the photo meta-data tag.

```console
$ ml classify cvbp images/*.png | 
  awk '$1>0.75{print}' |
  cut -d, -f2,3 | 
  tr ',' ' ' | 
  xargs -d'\n' -n1 bash -c 'mogrify -comment $0 $1'
```

## Demonstration

```console
$ ml demo cvbp
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
