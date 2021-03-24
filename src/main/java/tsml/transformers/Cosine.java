/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.transformers;

import java.io.File;
import java.io.IOException;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.*;

/*
     * copyright: Anthony Bagnall
     * @author Aaron Bostrom
 * 
 * */
public class Cosine implements Transformer {


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //multidimensional cosine. Cosine applied series wise for each dimension
        double[][] out = new double[inst.getNumDimensions()][];
        int index = 0;
        for(TimeSeries ts : inst){
            double[] data = new double[ts.getSeriesLength()];
            double n = data.length;
            for (int k = 0; k < n; k++) {
                double fk = 0;
                for (int i = 0; i < n; i++) {
                    double c = k * (i + 0.5) * (Math.PI / n);
                    fk += ts.getValue(i) * Math.cos(c);
                }
                data[k] = fk;
            }
            out[index++] = data;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex());
    }

    @Override
    public Instance transform(Instance inst) {
        int n = inst.numAttributes() - 1;
        Instance newInst = new DenseInstance(inst.numAttributes());
        for (int k = 0; k < n; k++) {
            double fk = 0;
            for (int i = 0; i < n; i++) {
                double c = k * (i + 0.5) * (Math.PI / n);
                fk += inst.value(i) * Math.cos(c);
            }
            newInst.setValue(k, fk);
        }

        // overrided cosine class value, with original.
        if (inst.classIndex() >= 0)
            newInst.setValue(inst.classIndex(), inst.classValue());
        return newInst;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        FastVector<Attribute> atts = new FastVector<>();

        for (int i = 0; i < inputFormat.numAttributes() - 1; i++) {
            // Add to attribute list
            String name = "Cosine_" + i;
            atts.addElement(new Attribute(name));
        }
        // Get the class values as a fast vector
        Attribute target = inputFormat.attribute(inputFormat.classIndex());

        FastVector<String> vals = new FastVector<>(target.numValues());
        for (int i = 0; i < target.numValues(); i++)
            vals.addElement(target.value(i));

        atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

        Instances result = new Instances("COSINE" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }

        System.out.println(result);

        return result;
    }

    public static void main(String[] args) throws IOException {

        String localPath = "src/main/java/experiments/data/tsc/"; //  path for testing.
        String datasetName = "ChinaTown";
        Instances train = DatasetLoading
                .loadData(localPath + datasetName + File.separator + datasetName + "_TRAIN.ts");
        Instances test = DatasetLoading
                .loadData(localPath + datasetName + File.separator + datasetName + "_TEST.ts");
        Cosine cosTransform = new Cosine();
        Instances out_train = cosTransform.transform(train);
        Instances out_test = cosTransform.transform(test);
        System.out.println(out_train.toString());
        System.out.println(out_test.toString());

    }



}
