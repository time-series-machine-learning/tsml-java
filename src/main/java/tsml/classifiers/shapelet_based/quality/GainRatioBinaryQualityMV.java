package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GainRatioBinaryQualityMV extends ShapeletQualityMV {

    public GainRatioBinaryQualityMV(double[][][] instancesArray,
                                    int[] classIndexes,
                                    String[] classNames,
                                    int[] classCounts,
                                    ShapeletDistanceMV distance){
        super();
        this.instancesArray = instancesArray;
        this.classIndexes = classIndexes;
        this.classNames = classNames;
        this.classCounts = classCounts;
        this.distance = distance;
        this.atts = new ArrayList<Attribute>(2);


        this.atts.add(new Attribute("distance"));

        FastVector my_nominal_values = new FastVector(2);
        my_nominal_values.addElement("positive");
        my_nominal_values.addElement("negative");

        this.atts.add(new Attribute("class",my_nominal_values));
    }

    @Override
    public double calculate(ShapeletMV candidate) {

        this.instances = new Instances("TestInstances",atts,0);
        this.instances.setClassIndex(1);

        double cl = 0;
        for (int i=0;i< instancesArray.length;i++){
            if (classIndexes[i]==candidate.getClassIndex()){
                cl = 0;
            }else{
                cl = 1;
            }
            double dist = distance.calculate(candidate,instancesArray[i]);
            instances.add(new DenseInstance(1,new double[]{dist,cl}));

        }
       // System.out.println(instances);

        GainRatioAttributeEval eval = new   GainRatioAttributeEval();
            try {
                eval.buildEvaluator(instances);
                return eval.evaluateAttribute(0);
            } catch (Exception e) {
                e.printStackTrace();

            }

            return 0;


    }


}
