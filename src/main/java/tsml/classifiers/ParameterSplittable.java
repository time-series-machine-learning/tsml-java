/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers;

import java.io.Serializable;

/**
 *
 * @author ajb
 */
public interface ParameterSplittable extends Serializable{
    public void setParamSearch(boolean b);
/* The actual parameter values should be set internally. This integer
  is just a key to maintain different parameter sets. The range starts at 1
    */
    public void setParametersFromIndex(int x);
//    public String getParas();
}
