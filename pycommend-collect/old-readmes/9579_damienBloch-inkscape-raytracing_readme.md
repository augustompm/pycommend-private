<img  src="https://github.com/damienBloch/inkscape-raytracing/blob/master/pictures/logo.jpeg" height="170" align="right"/>

# Inkscape Ray Optics

An extension for Inkscape that makes it easier to draw optical diagrams. 

Allows to annotate Inkscape primitives with optical properties and draws beam paths by taking into account reflection and refraction. 

[Bug reports](https://github.com/damienBloch/inkscape-raytracing/issues) or [suggestions](https://github.com/damienBloch/inkscape-raytracing/discussions) are welcome.

## Examples


| <img src="./pictures/sphere.svg"  width="600"> |
|:--:|
| *Beams through a sphere* |


| <img src="./pictures/blue_table.png"  width="600"> |
|:--:|
| *Optical table planning* |

## How to install 

- Download the zip folder containing the extension source code (Code > Download ZIP).  
- Extract the content of the ZIP folder in Inkscape user extensions directory.<br />The location of the extensions directory can be found in Inkscape with Edit > Preferences > System > User extensions.
- Restart Inkscape if open.

For Linux users this can typically be done with:
  ```shell
  cd ~/.config/inkscape/extensions
  git clone https://github.com/damienBloch/inkscape-raytracing
  ```
  
## Requirements
 
 * [Inkscape 1.2](https://inkscape.org/release/) or above.<br />The extension was not tested with older versions of Inkscape.
  
In addition, this extension also requires the following programs, but they are usually already installed with Inkscape:
  
   * [Python 3.9](https://www.python.org/downloads/) or above  
   * [NumPy](https://numpy.org/install/)
   * [Inkex](https://pypi.org/project/inkex/) 

If the version of python used by Inkscape doesn't satisfy the requirement, the interpreter can be changed by following these [instructions](https://wiki.inkscape.org/wiki/Extension_Interpreters#Selecting_a_specific_interpreter_version_.28via_preferences_file.29).
  

## How to use 

### 1. For each optical element or group of elements, select it and choose its material with `Extensions > Optics > Set material as...`:

<img src="./pictures/ray_tracing_1.png"  width="800">

The material can be one of the following:

  * `Beam`: source of the ray. Need at least one element with this property to see an effect. Typically the element should be a straight line.
  * `Mirror`: reflects an incoming beam. Element can be a closed or open shape.
  * `Beam dump`: absorbs all incoming beams. Element can be a closed or open shape.
  * `Beam splitter`: for each incoming beam, produces one transmitted beam and one reflected beam. Element can be a closed or open shape, but closed shape will cause the number of beams to increase exponentially.
  * `Glass`: with optical index. Transmits and bends a beam depending on its optical index. **Element must be a closed shape**.

This will automatically write some text in the element description. This text is used to reccord the properties of the elements. It is also possible to directly write the text in the description.
  
An element can have at most one optical property and will be ignored if it has two or more.

It is possible to add complementary text in the description. If it doesn't have the syntax `optics:<something>`, the extra text will simply be ignored.



### 2. Select the elements to render and run the extension with `Extensions > Optics > Ray Tracing`:

<img src="./pictures/ray_tracing_2.png"  width="800">

### 3. This will trace all the beams originated from a beam element:

<img src="./pictures/ray_tracing_3.png"  width="800">

The beams are added to a new sub-layer `generated_beams` of their parent layer.

Note that the borders of the document blocks the beams and all objects outside the document page will be ignored.


## Tips

* Using `Extensions > Optics > Lens...` adds a lens to the document with the right radius of curvature to get the desired focal length.
* For frequent use, it is possible to bind an extension to a hotkey with `Edit > Preferences > Interface > Keyboard Shortcuts > Extensions`. 
* This extension is compatible with clone objects (`Edit > Clone`). They are symbolic clones that mirrors all changes applied to the original object. 


## Known limitations

* Cannot write the properties in a group description. They must be written in the primitives description. 
* Avoid overlapping or touching elements. It won't cause Inkscape to crash, but might give unexpected results.
* The same goes for self-intersecting paths.
* Text elements are ignored whatever their description. If they need to be considered, they must be converted to path first.
