package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import weka.attributeSelection.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public class WekaQuality extends ShapeletQualityFunction {

    protected ASEvaluation eval;
    protected TransformedInstances transformedInstances;
    protected ArrayList<Attribute> atts;

    public WekaQuality(WekaEvaluatorFactory eval,
                       TransformedInstancesFactory transformedInstances,
                       TimeSeriesInstances instances){
       super(instances);
       this.eval = eval.createEvaluatorFactory();
       this.transformedInstances = transformedInstances.createTransformedInstances();
       this.atts = new ArrayList<Attribute>(2);
        atts.add(new Attribute("distance"));
        atts.add(new Attribute("class", Arrays.asList(this.transformedInstances.getClassLabels(instances))));
    }

    @Override
    public double calculate(ShapeletFunctions fun, ShapeletMV candidate) {

        Instances instances = new Instances("TestInstances",atts,0);
        instances.setClassIndex(instances.attribute("class").index());

        for (int i=0;i< trainInstances.numInstances();i++){
            double dist = fun.sDist(candidate,trainInstances.get(i));
            double classIndex = this.transformedInstances.getClassIndex(trainInstances.get(i).getTargetValue(),candidate.getClassIndex());
            instances.add(new DenseInstance(1,new double[]{dist,classIndex}));

        }


            try {
               eval.buildEvaluator(instances);

                return ((AttributeEvaluator)eval).evaluateAttribute(0);
            }catch (IllegalArgumentException e) {
               // e.printStackTrace();
                return 0;

            }
            catch (Exception e) {
                e.printStackTrace();

            }

            return 0;


    }

    interface TransformedInstances{
        double getClassIndex(double classIndex, double candiateClassIndex);
        String[] getClassLabels(TimeSeriesInstances instances);
    }

    static class BinaryTransformedInstances implements TransformedInstances{

        @Override
        public double getClassIndex(double classIndex, double candiateClassIndex) {
            return classIndex==candiateClassIndex?1.0:0.0;
        }

        @Override
        public String[] getClassLabels(TimeSeriesInstances instances) {
            return new String[]{"class0","class1"};
        }
    }

    static class RegularTransformedInstances implements TransformedInstances{

        @Override
        public double getClassIndex(double classIndex, double candiateClassIndex) {
            return classIndex;
        }

        @Override
        public String[] getClassLabels(TimeSeriesInstances instances) {
            return instances.getClassLabels();
        }
    }

    public enum TransformedInstancesFactory {
        REGULAR {
            @Override
            public TransformedInstances createTransformedInstances() {
                return new RegularTransformedInstances();
            }
        },
        BINARY {
            @Override
            public TransformedInstances createTransformedInstances() {
                return new BinaryTransformedInstances();
            }
        };

        public abstract TransformedInstances createTransformedInstances() ;
    }

    public enum WekaEvaluatorFactory {
        GAIN {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new GainRatioAttributeEval();
            }
        },
        CHI {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new ChiSquaredAttributeEval();
            }
        },
        CORRELATION {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new  InfoGainAttributeEval();
            }
        },
        INFO_GAIN {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new  InfoGainAttributeEval();
            }
        },
        FSTAT {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new ReliefFAttributeEval();
            }
        },
        ONE_R {
            @Override
            public ASEvaluation createEvaluatorFactory() {
                return new  OneRAttributeEval();
            }
        };

        public abstract ASEvaluation createEvaluatorFactory() ;
    }


}
