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
 * Takes the log of all values in the series
 *
 * */
public class Log implements Transformer {


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //multidimensional log. Log applied series wise for each dimension
        double[][] out = new double[inst.getNumDimensions()][];
        int index = 0;
        for(TimeSeries ts : inst){
            double[] data = new double[ts.getSeriesLength()];
            double n = data.length;
            for (int i = 0; i < n; i++) {
                data[i] = Math.log(ts.get(i));
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
            newInst.setValue(i,  Math.log(inst.value(i)));
        }

        // overrided log class value, with original.
        if (inst.classIndex() >= 0)
            newInst.setValue(inst.classIndex(), inst.classValue());
        return newInst;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        FastVector<Attribute> atts = new FastVector<>();

        for (int i = 0; i < inputFormat.numAttributes() - 1; i++) {
            // Add to attribute list
            String name = "Log_" + i;
            atts.addElement(new Attribute(name));
        }
        // Get the class values as a fast vector
        Attribute target = inputFormat.attribute(inputFormat.classIndex());

        FastVector<String> vals = new FastVector<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++)
            vals.addElement(target.value(i));

        atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

        Instances result = new Instances("LOG" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }

        System.out.println(result);

        return result;
    }

    public static void main(String[] args) throws Exception {
        // final double[][] t1 = {{0, Math.PI, Math.PI*2},{ Math.PI * 0.5, Math.PI *
        // 1.5, Math.PI*2.5}};
        // final double[] labels = {1,2};
        // final Instances train = InstanceTools.toWekaInstances(t1, labels);

        Instances[] data = DatasetLoading.sampleItalyPowerDemand(0);
        Log logTransform = new Log();
        Instances out_train = logTransform.transform(data[0]);
        Instances out_test = logTransform.transform(data[1]);
        System.out.println(out_train.toString());
        System.out.println(out_test.toString());

    }



}
