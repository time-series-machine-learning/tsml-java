package tsml.classifiers.shapelet_based.dev.classifiers;

import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.SingleTestSetEvaluatorTS;
import experiments.Experiments;
import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.ensembles.legacy.EnhancedRotationForest;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.shapelet_based.dev.filter.*;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctionsDependant;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctionsDimDependant;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctionsIndependent;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.quality.WekaQuality;
import tsml.classifiers.shapelet_based.dev.quality.WekaQualityFiltered;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.transformers.shapelet_tools.Shapelet;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class MSTC extends EnhancedAbstractClassifier {

    private ShapeletParams params;
    private Classifier classifier;
    private List<ShapeletMV> shapelets;
    private Instances transformData;

    public static Random RAND = new Random();


    public ShapeletFilterMV filter;

    public MSTC(ShapeletParams params){
        //super(numClasses);
        this.params = params;
        this.estimateOwnPerformance = true;
        this.ableToEstimateOwnPerformance = true;
    }


    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {




        filter =  this.params.filter.createFilter();
        filter.setHourLimit(this.params.contractTimeHours);
        shapelets =filter.findShapelets(params, data);
        transformData = buildTansformedDataset(data);

        classifier = params.classifier.createClassifier();
        classifier.buildClassifier(transformData);
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator(); //DONT clone data, DO set the class to be missing for each inst

        this.trainResults =  eval.evaluate(this, data);
        this.trainResults.setParas(getParameters());
    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance data) throws Exception {
        Instance transformData = buildTansformedInstance(data);
        return classifier.distributionForInstance(transformData);
    }

    @Override
    public double classifyInstance(TimeSeriesInstance data) throws Exception {
        Instance transformData = buildTansformedInstance(data);
        return classifier.classifyInstance(transformData);
    }

    private Instances buildTansformedDataset(TimeSeriesInstances instances) {
        Instances output = determineOutputFormat(instances);
        int size = shapelets.size();
        int dataSize = instances.numInstances();

        for (int j = 0; j < dataSize; j++) {
            output.add(new DenseInstance(size + 1));
        }
        ShapeletFunctions fun = params.type.createShapeletType();
        double dist;
        for (int j = 0; j < dataSize; j++) {
            int i=0;
            for (ShapeletMV shapelet: this.shapelets) {
                dist = fun.sDist( shapelet,instances.get(j));
                output.instance(j).setValue(i, dist);
                i++;
            }
        }


        for (int j = 0; j < dataSize; j++) {
            output.instance(j).setValue(size, instances.get(j).getTargetValue());
        }

        return output;
    }

    private Instance buildTansformedInstance(TimeSeriesInstance instance) {
        Shapelet s;
        int size = shapelets.size();
        Instance out = new DenseInstance(size + 1);
        ShapeletFunctions fun = params.type.createShapeletType();
        out.setDataset(transformData);
        double dist;
        int i=0;
        for (ShapeletMV shapelet: this.shapelets) {
            dist = fun.sDist(shapelet,instance);
            out.setValue(i, dist);

            i++;
        }
        return out;
    }

    private Instances determineOutputFormat(TimeSeriesInstances inputFormat) throws IllegalArgumentException {

        if (this.shapelets.size() < 1) {
            throw new IllegalArgumentException("ShapeletTransform not initialised correctly - " +
                    "please specify a value of k (this.numShapelets) that is greater than or equal to 1. " +
                    "It is currently set tp "+this.shapelets.size());
        }

        int length = this.shapelets.size();
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < length; i++) {
            name = "Shapelet_" + i;
            Attribute att = new Attribute(name);
            att.setWeight(this.shapelets.get(i).getQuality());
            atts.add(att);
        }

        FastVector vals = new FastVector(inputFormat.numClasses());
        for (int i = 0; i < inputFormat.numClasses(); i++) {
            vals.addElement(inputFormat.getClassLabels()[i]);
        }
        atts.add(new Attribute("Target", vals));

        Instances result = new Instances("Shapelets" + inputFormat.getProblemName(), atts, inputFormat.numInstances());
        result.setClassIndex(result.numAttributes() - 1);
        return result;
    }

    @Override
    public String getParameters(){
        return filter.getParameters();
    }



    public enum ShapeletFilters {
        EXHAUSTIVE {
            @Override
            public ShapeletFilterMV createFilter() {
                return new ExhaustiveFilter();
            }
        },
        EXHAUSTIVE_BY_CLASS {
            @Override
            public ShapeletFilterMV createFilter() {
                return new ExhaustiveFilter();
            }
        },
        RANDOM {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilter();
            }
        },
        RANDOM_BY_SERIES {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilterBySeries();
            }
        },
        RANDOM_BY_CLASS {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilterByClass();
            }
        };;

        public abstract ShapeletFilterMV createFilter();
    }

    public enum ShapeletQualities {
        GAIN {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.GAIN,
                        WekaQuality.TransformedInstancesFactory.REGULAR,
                        instances);
            }
        },
        GAIN_BINARY {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.GAIN,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
        },
        GAIN_BINARY_FILTERED {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQualityFiltered(
                        WekaQuality.WekaEvaluatorFactory.GAIN,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
        },

        /*     ORDER_LINE {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new OrderlineQuality(instances);
            }
        },
        ORDER_LINE_BINARY {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new OrderlineBinaryQuality(instances);
            }
        },

        */
       CORR{
           public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
               return new WekaQuality(
                       WekaQuality.WekaEvaluatorFactory.CORRELATION,
                       WekaQuality.TransformedInstancesFactory.REGULAR,
                       instances);
           }
       },
        CORR_BINARY{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.CORRELATION,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
         },
        ONE_R{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.ONE_R,
                        WekaQuality.TransformedInstancesFactory.REGULAR,
                        instances);
            }
        },
        ONE_R_BINARY{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.ONE_R,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
        },
        FSTAT{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.FSTAT,
                        WekaQuality.TransformedInstancesFactory.REGULAR,
                        instances);
            }
        },
        FSTAT_BINARY{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.FSTAT,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
        },

        CHI{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.CHI,
                        WekaQuality.TransformedInstancesFactory.REGULAR,
                        instances);
            }
        },

        CHI_BINARY{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) {
                return new WekaQuality(
                        WekaQuality.WekaEvaluatorFactory.CHI,
                        WekaQuality.TransformedInstancesFactory.BINARY,
                        instances);
            }
        };

        public abstract ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances) ;
    }



    public enum ShapeletFactories {
        DEPENDANT {
            @Override
            public ShapeletFunctions createShapeletType() {
                return new ShapeletFunctionsDependant();
            }
        },
        DIMENSION_DEPENDANT {
            @Override
            public ShapeletFunctions createShapeletType() {
                return new ShapeletFunctionsDimDependant();
            }
        },

        INDEPENDENT {
            @Override
            public ShapeletFunctions createShapeletType() {
                return new ShapeletFunctionsIndependent();
            }
        };

        public abstract ShapeletFunctions createShapeletType();
    }

    public enum AuxClassifiers {
        ENSEMBLE {
            @Override
            public Classifier createClassifier() {
                CAWPE ensemble=new CAWPE();
                ensemble.setWeightingScheme(new TrainAcc(4));
                ensemble.setVotingScheme(new MajorityConfidence());
                Classifier[] classifiers = new Classifier[7];
                String[] classifierNames = new String[7];

                SMO smo = new SMO();
                smo.turnChecksOff();
                smo.setBuildLogisticModels(true);
                PolyKernel kl = new PolyKernel();
                kl.setExponent(2);
                smo.setKernel(kl);
                classifiers[0] = smo;
                classifierNames[0] = "SVMQ";

                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                classifiers[1] = r;
                classifierNames[1] = "RandF";


                RotationForest rf=new RotationForest();
                rf.setNumIterations(100);
                classifiers[2] = rf;
                classifierNames[2] = "RotF";
                IBk nn=new IBk();
                classifiers[3] = nn;
                classifierNames[3] = "NN";
                NaiveBayes nb=new NaiveBayes();
                classifiers[4] = nb;
                classifierNames[4] = "NB";
                J48 c45=new J48();
                classifiers[5] = c45;
                classifierNames[5] = "C45";
                SMO svml = new SMO();
                svml.turnChecksOff();
                svml.setBuildLogisticModels(true);
                PolyKernel k2 = new PolyKernel();
                k2.setExponent(1);
                smo.setKernel(k2);
                classifiers[6] = svml;
                classifierNames[6] = "SVML";
                ensemble.setClassifiers(classifiers, classifierNames, null);

                return  ensemble;
            }
        },
        LINEAR {
            @Override
            public Classifier createClassifier() {
                return new LibLinearTS();

            }
        },
        SVM {
            @Override
            public Classifier createClassifier() {
                SMO svml = new SMO();
                svml.turnChecksOff();
                svml.setBuildLogisticModels(true);
                PolyKernel k2 = new PolyKernel();
                k2.setExponent(1);
                svml.setKernel(k2);
                return  svml;

            }
        },
        ROT{
            @Override
            public Classifier createClassifier() {
                EnhancedRotationForest rotf=new EnhancedRotationForest();
                rotf.setMaxNumTrees(200);
                rotf.setTrainTimeLimit(TimeUnit.HOURS,10);
                return rotf;
            }
        },
        ROT_2H{
            @Override
            public Classifier createClassifier() {
                EnhancedRotationForest rotf=new EnhancedRotationForest();
                rotf.setMaxNumTrees(200);
                rotf.setTrainTimeLimit(TimeUnit.HOURS,2);
                return rotf;
            }
        },
        NB{
            @Override
            public Classifier createClassifier() {

                return new NaiveBayes();
            }
        },
        FILTERED{
            @Override
            public Classifier createClassifier() {
                FilteredClassifier fc = new FilteredClassifier();
                LibLinearTS ll = new LibLinearTS();
                fc.setClassifier( ll);
                return fc;
            }
        }
        ;

        public abstract Classifier createClassifier();
    }

    public static class ShapeletParams{
        public int k;
        public int min;
        public int max;
        public int maxIterations;
        public int contractTimeHours;
        public boolean allowZeroQuality = false;
        public boolean removeSelfSimilar = true;

        public ShapeletFilters filter;
        public ShapeletQualities quality;
        public ShapeletFactories type;
        public AuxClassifiers classifier;

        public ShapeletParams(int k, int min, int max, int maxIterations, int contractTimeHours,
                              ShapeletFilters filter, ShapeletQualities quality,
                              ShapeletFactories type,
                              AuxClassifiers classifier){
            this.k = k ;
            this.min = min;
            this.max = max;
            this.maxIterations = maxIterations;
            this.contractTimeHours = contractTimeHours;
            this.filter = filter;
            this.quality = quality;
            this.type = type;
            this.classifier = classifier;
            this.allowZeroQuality = false;
            this.removeSelfSimilar = true;
        }

        public ShapeletParams(ShapeletParams params){
            this.k = params.k ;
            this.min = params.min;
            this.max = params.max;
            this.maxIterations = params.maxIterations;
            this.contractTimeHours = params.contractTimeHours;
            this.filter = params.filter;
            this.quality = params.quality;
            this.type = params.type;
            this.classifier = params.classifier;
            this.allowZeroQuality = params.allowZeroQuality;
            this.removeSelfSimilar = params.removeSelfSimilar;
        }

    }

    public static void main(String[] arg){
        String m_local_path = "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\";

        String dataset = "BasicMotions";
        String filepath = m_local_path + dataset + "\\" + dataset;

        TSReader ts_reader = null;
        try {
            ts_reader = new TSReader(new FileReader(new File(filepath + "_TRAIN" + ".ts")));
            TimeSeriesInstances ts_train_data = ts_reader.GetInstances();

            ts_reader = new TSReader(new FileReader(new File(filepath + "_TEST" + ".ts")));
            TimeSeriesInstances ts_test_data = ts_reader.GetInstances();
            int f = ts_train_data.numInstances()*ts_train_data.getMaxLength()*ts_train_data.getMaxNumDimensions();
            System.out.println( "f= " + f);
            ShapeletParams params = new ShapeletParams(f,3,ts_train_data.getMaxLength()/2,
                    10000,4,
                    ShapeletFilters.RANDOM, ShapeletQualities.GAIN,
                    ShapeletFactories.DEPENDANT,
                    AuxClassifiers.ENSEMBLE);

            MSTC shapelet = new MSTC(  params);
            shapelet.buildClassifier(ts_train_data);

            double ok=0, wrong=0;
            for (TimeSeriesInstance ts: ts_test_data){
                double pred = shapelet.classifyInstance(ts);
                if (ts.getTargetValue()==pred){
                    ok++;
                }else{
                    wrong++;
                }
            }
            System.out.println("Acc= " + ok/(ok+wrong));


        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }
}
