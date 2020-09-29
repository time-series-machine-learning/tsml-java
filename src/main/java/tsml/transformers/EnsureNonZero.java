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
package tsml.transformers;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.*;

/*
 * copyright: Anthony Bagnall
 * @author James Large
 *
 * Replaces zero-values in the series with Double.MIN_VALUE
 *
 * */
public class EnsureNonZero implements Transformer {


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //multidimensional EnsureNonZero. EnsureNonZero applied series wise for each dimension
        double[][] out = new double[inst.getNumDimensions()][];
        int index = 0;
        for(TimeSeries ts : inst){
            double[] data = new double[ts.getSeriesLength()];
            double n = data.length;
            for (int i = 0; i < n; i++) {
                double newval = ts.get(i) == 0.0 ? Double.MIN_VALUE : ts.get(i);
                data[i] = newval;
            }
            out[index++] = data;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex());
    }

    @Override
    public Instance transform(Instance inst) {
        int n = inst.numAttributes() - 1;
        Instance newInst = new DenseInstance(inst.numAttributes());
        for (int i = 0; i < n; i++) {
            double newval = inst.value(i) == 0.0 ? Double.MIN_VALUE : inst.value(i);
            newInst.setValue(i,  newval);
        }

        // overrided EnsureNonZero class value, with original.
        if (inst.classIndex() >= 0)
            newInst.setValue(inst.classIndex(), inst.classValue());
        return newInst;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        FastVector<Attribute> atts = new FastVector<>();

        for (int i = 0; i < inputFormat.numAttributes() - 1; i++) {
            // Add to attribute list
            String name = "EnsureNonZero_" + i;
            atts.addElement(new Attribute(name));
        }
        // Get the class values as a fast vector
        Attribute target = inputFormat.attribute(inputFormat.classIndex());

        FastVector<String> vals = new FastVector<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++)
            vals.addElement(target.value(i));

        atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

        Instances result = new Instances("EnsureNonZero" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }

        System.out.println(result);

        return result;
    }

    public static void main(String[] args) throws Exception {

        Instances[] data = DatasetLoading.sampleItalyPowerDemand(0);
        EnsureNonZero zeroTransform = new EnsureNonZero();
        Instances out_train = zeroTransform.transform(data[0]);
        Instances out_test = zeroTransform.transform(data[1]);
        System.out.println(out_train.toString());
        System.out.println(out_test.toString());

    }



}
