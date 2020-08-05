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
package tsml.transformers.shapelet_tools.class_value;

import java.io.Serializable;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.class_counts.ClassCounts;
import utilities.class_counts.TreeSetClassCounts;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class NormalClassValue implements Serializable{
    
    double shapeletValue;
    ClassCounts classDistributions;
    
    public void init(Instances inst)
    {
        classDistributions = new TreeSetClassCounts(inst);
    }

    public void init(TimeSeriesInstances inst)
    {
        classDistributions = new TreeSetClassCounts(inst);
    }
    
    public ClassCounts getClassDistributions()
    {
        return classDistributions;
    }
    
    //this will get updated as and when we work with a new shapelet.
    public void setShapeletValue(Instance shapeletSeries)
    {
        shapeletValue = shapeletSeries.classValue();
    }

    //this will get updated as and when we work with a new shapelet.
    public void setShapeletValue(TimeSeriesInstance shapeletSeries)
    {
        shapeletValue = shapeletSeries.getLabelIndex();
    }
    
    public double getClassValue(Instance in){
        return in.classValue();
    }

    public double getClassValue(TimeSeriesInstance in){
        return in.getLabelIndex();
    }

    public double getShapeletValue() {
        return shapeletValue;
    }
    
}
