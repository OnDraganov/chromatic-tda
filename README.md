# chromatic_tda

`chromatic_tda` is a package for computing six-packs of persistent diagrams of colored point clouds. Currently point clouds in R^2 with 2 or 3 colors are supported.

# Installation

## Install with `pip`

The package is uploaded to [PyPI](https://pypi.org/project/chromatic-tda/), so it can be installed with `pip`:
Run `pip install chromatic_tda`.

## Install from github project

The project uses `poetry` for Python dependecy management, which allows you to easily install all you need to run the package:

- Clone the repository
- Install `poetry`
- Go to the repository folder in terminal
- Run `poetry shell`
- Run `poetry install`
- [optional] Run `pytest` to check that the code runs

# How to use

Check the jupyter notebook file `manual` to see how to use the code.

# Future

The code is under active developement. The future plans include:
- Adding support for points in R^3
- Making persistence computation faster with clearing.

# License

Copyright ©2023. Institute of Science and Technology Austria (IST Austria). All Rights Reserved.  

This file is part of chromatic_tda, which is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
 
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 
You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 
Contact the Technology Transfer Office, ISTA, Am Campus 1, A-3400 Klosterneuburg, Austria, +43-(0)2243 9000, twist@ist.ac.at, for commercial licensing opportunities.