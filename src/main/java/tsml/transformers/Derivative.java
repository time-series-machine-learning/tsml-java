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

import java.io.Serializable;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: class to take the derivative of a time series.
 * <p>
 * Contributors: goastler, Jason Lines
 */
public class Derivative implements Transformer, Serializable {

    private static Derivative INSTANCE;
    // Global derivative function which is cached, i.e. if you ask it to convert the
    // same instance twice it will
    // instead fetch from the cache the second time
    private static CachedTransformer GLOBAL_CACHE;

    public static Derivative getGlobalInstance() {
        if (INSTANCE == null) {
            INSTANCE = new Derivative();
        }
        return INSTANCE;
    }

    public static CachedTransformer getGlobalCachedTransformer() {
        if (GLOBAL_CACHE == null) {
            GLOBAL_CACHE = new CachedTransformer(getGlobalInstance());
        }
        return GLOBAL_CACHE;
    }

    public static double[] getDerivative(double[] input, boolean classValOn) {

        int classPenalty = 0;
        if (classValOn) {
            classPenalty = 1;
        }

        double[] derivative = new double[input.length];

        for (int i = 1; i < input.length - 1 - classPenalty; i++) { // avoids class Val if present
            derivative[i] = ((input[i] - input[i - 1]) + ((input[i + 1] - input[i - 1]) / 2)) / 2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length - 1 - classPenalty] = derivative[derivative.length - 2 - classPenalty];
        if (classValOn) {
            derivative[derivative.length - 1] = input[input.length - 1];
        }
        return derivative;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        //If the class index exists.
        if(inputFormat.classIndex() >= 0) {
            if (inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
                throw new IllegalArgumentException("cannot handle class values not at end");
            }
        }
        return new Instances(inputFormat, inputFormat.size());
    }

    @Override
    public Instance transform(Instance inst) {
        final double[] derivative = getDerivative(inst.toDoubleArray(), true);
        final Instance copy = new DenseInstance(inst.weight(), derivative);
        copy.setDataset(inst.dataset());
        return copy;                                                                      // careful!
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for (TimeSeries ts : inst) {
            out[i++] = getDerivative(ts.toValueArray(), false);
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }


}
