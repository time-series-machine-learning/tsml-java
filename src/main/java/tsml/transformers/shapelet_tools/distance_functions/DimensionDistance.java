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
package tsml.transformers.shapelet_tools.distance_functions;

import java.io.Serializable;

import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class DimensionDistance extends ImprovedOnlineShapeletDistance implements Serializable{
    
    @Override
    public double calculate(Instance timeSeries, int timeSeriesId){
        //split the timeSeries up and pass in the specific shapelet dim.
        Instance[] dimensions = utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance(timeSeries);
        return calculate(dimensions[dimension].toDoubleArray(), timeSeriesId);
    }
    
}
