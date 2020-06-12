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
package tsml.filters;

import static experiments.data.DatasetLoading.sampleGunPoint;
import static utilities.Utilities.extractTimeSeries;

import java.io.Serializable;
import org.junit.Assert;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * Purpose: class to take the derivative of a time series.
 * <p>
 * Contributors: goastler, Jason Lines
 */
public class Derivative
    extends SimpleBatchFilter implements Serializable {

    private static Derivative INSTANCE;
    // Global derivative function which is cached, i.e. if you ask it to convert the same instance twice it will
    // instead fetch from the cache the second time
    private static CachedFilter GLOBAL_CACHE;

    // prefix for dataset name
    public static String getPrefix() {
        return "der_";
    }

    public static Derivative getGlobalInstance() {
        if(INSTANCE == null) {
            INSTANCE = new Derivative();
        }
        return INSTANCE;
    }

    public static CachedFilter getGlobalCache() {
        if(GLOBAL_CACHE == null) {
            GLOBAL_CACHE = new CachedFilter(getGlobalInstance());
        }
        return GLOBAL_CACHE;
    }

    public static Instance getDerivative(Instance instance) {
        instance = (Instance) instance.copy();
        double[] derivative = getDerivative(instance.toDoubleArray(), true);
        for(int i = 0; i < derivative.length; i++) {
            instance.setValue(i, derivative[i]);
        }
        return instance;
    }

    public static double[] getDerivative(double[] input, boolean classValOn) {

        int classPenalty = 0;
        if(classValOn) {
            classPenalty = 1;
        }

        double[] derivative = new double[input.length];

        for(int i = 1; i < input.length - 1 - classPenalty; i++) { // avoids class Val if present
            derivative[i] = ((input[i] - input[i - 1]) + ((input[i + 1] - input[i - 1]) / 2)) / 2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length - 1 - classPenalty] = derivative[derivative.length - 2 - classPenalty];
        if(classValOn) {
            derivative[derivative.length - 1] = input[input.length - 1];
        }
        return derivative;
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        return super.setInputFormat(instanceInfo);
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return new Instances(inputFormat, 0);
    }

    public Instances process(Instances data) throws Exception {
        Assert.assertEquals(data.numAttributes() - 1, data.classIndex());
        
        Instances output = determineOutputFormat(data);

        // for each data, get distance to each shapelet and create new instance
        for(int i = 0; i < data.numInstances(); i++) { // for each data
            Instance instance = data.get(i);
            instance = getDerivative(instance);
            output.add(instance);
        }
        return output;
    }

    public static void main(String[] args) {
        try {
            Instances instances = sampleGunPoint(0)[0];
            Instances processed = new Derivative().process(instances);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

}
