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

import utilities.serialisation.SerialisedFunction;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.*;

import java.io.Serializable;


public class Derivative
    extends SimpleBatchFilter implements Serializable {

    public static final String PREFIX = "der_";

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        if(inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
            throw new IllegalArgumentException("cannot handle class values not at end");
        }
        inputFormat = new Instances(inputFormat, 0);
        for(int i = 0; i < inputFormat.numAttributes(); i++) {
            if(i != inputFormat.classIndex()) {
                inputFormat.renameAttribute(i, PREFIX + inputFormat.attribute(i).name());
            }
        }
        inputFormat.setRelationName(PREFIX + inputFormat.relationName());
        return inputFormat;

//        int numAttributes = inputFormat.numAttributes();
//
//        FastVector atts = new FastVector();
//        String name;
//        for(int i = 0; i < numAttributes-1; i++){
//            name = "Attribute_" + i;
//            atts.addElement(new Attribute(name));
//        }
//
//        if(inputFormat.classIndex() >= 0){ //Classification set, set class
//            //Get the class values as a fast vector
//            Attribute target = inputFormat.attribute(inputFormat.classIndex());
////
//            FastVector vals = new FastVector(target.numValues());
//            for(int i = 0; i < target.numValues(); i++){
//                vals.addElement(target.value(i));
//            }
//            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
//        }
//        Instances result = new Instances("derivativeTransform_" + inputFormat.relationName(), atts, inputFormat.numInstances());
//        if(inputFormat.classIndex() >= 0){
//            result.setClassIndex(result.numAttributes() - 1);
//        }
//
//        return result;
        
    }
    
    public Instances process(Instances data) throws Exception{

        Instances output = determineOutputFormat(data);

        // for each data, get distance to each shapelet and create new instance
        for(int i = 0; i < data.numInstances(); i++){ // for each data
            
            double[] rawData = data.instance(i).toDoubleArray();
            double[] derivative = getDerivative(rawData,true); // class value has now been removed - be careful!

            DenseInstance instance = new DenseInstance(1, derivative);
            output.add(instance);
        }
        return output;
    }
    
    
    public static double[] getDerivative(double[] input, boolean classValOn){

        int classPenalty = 0;
        if(classValOn){
            classPenalty=1;
        }

        double[] derivative = new double[input.length];

        for(int i = 1; i < input.length-1-classPenalty;i++){ // avoids class Val if present
            derivative[i] = ((input[i]-input[i-1])+((input[i+1]-input[i-1])/2))/2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length-1-classPenalty] = derivative[derivative.length-2-classPenalty];
        if(classValOn) {
            derivative[derivative.length - 1] = input[input.length - 1];
        }
        return derivative;
    }

    public static final Derivative INSTANCE = new Derivative();

    public static final SerialisedFunction<Instance, Instance> INSTANCE_DERIVATIVE_FUNCTION = instance -> {
        try {
            return Utilities.filter(instance, new Derivative());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    };


    public static final SerialisedFunction<Instances, Instances> INSTANCES_DERIVATIVE_FUNCTION = instances -> {
        try {
            return Utilities.filter(instances, new Derivative());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    };

}
