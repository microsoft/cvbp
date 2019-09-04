# Computer Vision Best Practice

This [MLHub](https://mlhub.ai) package provides a demo of Microsoft's
Computer Vision Best Practice repository available from
<https://github.com/microsoft/ComputerVision>, providing examples and
command line tools for computer vision tasks.  A collection of
pre-built Computer Vision models are exposed as command line tools
building on the notebook examples from the repository.

Visit the github repository for more details:
<https://github.com/microsoft/cvbp>

## Quick Start Examples

```console
$ ml demo     cvbp
$ ml tag      cvbp https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
$ ml classify cvbp
$ ml detect   cvbp
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

A demonstration of the package is presented below using the *demo*
command. The package also provides a number of useful command line
tools which we introduce here.

**tag**

The *tag* command will identify the dominant object in a photo with a
reported level of confidence. The confidence and tag and filename are
returned from the command, and can be piped on to other commands
within a command line to, for example, add the tag to the meta-data of
the image file.

This first example is classified 100% as a koala.
```console
$ ml tag cvbp https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
1.00,koala,https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG
```

Perhaps we would like to download this photo and then add the
appropriate tag to it:
```console
$ wget https://upload.wikimedia.org/wikipedia/commons/2/2d/Koala_in_Australia.JPG -O koala.jpg
$ ml tag cvbp koala.jpg |
  cut -d, -f2 |
  xargs bash -c 'mogrify -comment $0 koala.jpg' 
$ exiftool koala.jpg | grep -i comment
Comment                         : koala
```

Here we identify the primary tag for a collection of photos.

```console
$ ml tag cvbp image_*.png 
0.78,chickadee,image_01_bw_color.png
1.00,custard_apple,image_02_bw_color.png
0.98,echidna,image_03_bw_color.png
0.45,clumber,image_04_bw_color.png
1.00,beacon,image_05_bw_color.png
0.49,Tibetan_terrier,image_06_bw_color.png
0.90,great_white_shark,image_07_bw_color.png
0.65,bell_pepper,image_09_bw_color.png
0.87,redshank,image_10_bw_color.png
```

Perhaps we wish to add a tag to each photo in the current folder:
```console
$ ml tag cvbp *.jpg | 
  cut -d, -f2,3 | 
  tr ',' ' ' | 
  xargs -d'\n' -n1 bash -c 'mogrify -comment $0 $1'
```

**classify**

The *classify* command will open up the computer's webcam and begin
classifying the primary object within the frame of the camera.

```console
$ ml classify cvbp
```

**detect**

The *detect* command will open up the computer's webcam and begin
detecting objects within the frame of the camera.

```console
$ ml detect cvbp
```

# Demonstration

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
