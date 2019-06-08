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
package timeseriesweka.filters;

import weka.core.*;
import weka.filters.*;

import static utilities.Utilities.extractAttributesNoClassValue;


public class DerivativeFilter extends SimpleBatchFilter{

    public String globalInfo() {
        return "derivative";
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();
        result.enable(Capabilities.Capability.NO_CLASS); // filter doesn't need class to be set
        return result;
    }

    protected Instances determineOutputFormat(Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    public Instances process(Instances inst) {
        Instances result = new Instances(determineOutputFormat(inst), 0);
        for (int i = 0; i < inst.numInstances(); i++) {
            Instance instance = inst.get(i);
            double[] values = derivative(extractAttributesNoClassValue(instance));
            double[] valuesAndClass = new double[values.length + 1];
            System.arraycopy(values, 0, valuesAndClass, 0, values.length);
            valuesAndClass[valuesAndClass.length - 1] = instance.classValue();
            result.add(new DenseInstance(1, valuesAndClass));
        }
        return result;
    }

    public static double[] derivative(double[] input) {

        double[] derivative = new double[input.length];

        for(int i = 1; i < input.length-1;i++){
            derivative[i] = ((input[i]-input[i-1])+((input[i+1]-input[i-1])/2))/2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length-1] = derivative[derivative.length-2];

        return derivative;
    }

}
