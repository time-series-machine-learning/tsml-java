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
 
package tsml.classifiers;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
public interface TSClassifier{
    

    public Classifier getClassifier();
    public TimeSeriesInstances getTSTrainData();
    public void setTSTrainData(TimeSeriesInstances train);

    public default void buildClassifier(TimeSeriesInstances data) throws Exception{
        setTSTrainData(data);
        getClassifier().buildClassifier(Converter.toArff(data));
    }

    public default double[] distributionForInstance(TimeSeriesInstance inst) throws Exception{
        return getClassifier().distributionForInstance(Converter.toArff(inst, getTSTrainData().getClassLabels()));
    }

    public default double classifyInstance(TimeSeriesInstance inst) throws Exception{
        return getClassifier().classifyInstance(Converter.toArff(inst, getTSTrainData().getClassLabels()));
    }

    public default double[][] distributionForInstances(TimeSeriesInstances data) throws Exception {
        double[][] out = new double[data.numInstances()][];

        Instances data_inst = Converter.toArff(data);
        int i=0;
        for(Instance inst : data_inst)
            out[i++] = getClassifier().distributionForInstance(inst);

        return out;
    }

    public default double[] classifyInstances(TimeSeriesInstances data) throws Exception {
        double[] out = new double[data.numInstances()];
        Instances data_inst = Converter.toArff(data);
        int i=0;
        for(Instance inst : data_inst)
            out[i++] = getClassifier().classifyInstance(inst);
        return out;
    }
}