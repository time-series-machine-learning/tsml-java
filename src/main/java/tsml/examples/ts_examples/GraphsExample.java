package tsml.examples.ts_examples;

import java.util.Arrays;

import machine_learning.classifiers.kNN;
import tsml.classifiers.multivariate.ConcatenateClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import tsml.graphs.Pipeline;
import tsml.graphs.Pipeline.*;
import tsml.transformers.Cosine;
import tsml.transformers.Sine;
import tsml.transformers.Truncator;

public class GraphsExample {

    public static void example1() throws Exception {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0, 5.0 }, },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 3.0, 2.0, 1.0, 1.0 }, } };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 }, } };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("sine", new Sine());
        model.add("kNN", new kNN());

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }


    public static void example2() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { 
            {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 },
                {4.0,3.0,2.0,1.0, 7.0, 8.0}
            } 
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("Truncator", new Truncator()); //this chops the uneven off.
        model.concat("stack", 
            new Pipeline.TransformerLayer("Sine0", new Sine()), 
            new Pipeline.TransformerLayer("Cosine1", new Cosine())
        );

        //TODO: fix for MultivariateSingleEnsemble
        //model.add("kNN", new MultivariateSingleEnsemble(new kNN()));

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }

    public static void example3() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 },
                {4.0,3.0}, 
            } 
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("Truncator", new Truncator()); //this chops the uneven off.
        
        Pipeline model0 = new Pipeline();
        model0.add("Sine0", new Sine());
        model0.add("kNN", new kNN());

        Pipeline model1 = new Pipeline();
        model1.add("Cosine1", new Cosine());
        model1.add("kNN", new kNN());

        //this will use mean ensembling across the split pipelines.
        model.splitAndEnsemble("split", model0, model1);

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }


    public static void example4() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 },
                {4.0,3.0}, 
            } 
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("Truncator", new Truncator()); //this chops the uneven off.
        
        Pipeline model0 = new Pipeline();
        model0.add("Sine0", new Sine());

        Pipeline model1 = new Pipeline();
        model1.add("Cosine1", new Cosine());

        //if you're going to split and merge. 
        //unless you want to stack. don't put a classifier on the end of the pipelines.
        model.split("split", model0, model1);
        model.add("kNN", new ConcatenateClassifier(new kNN()));

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }


    public static void example5() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 },
                {4.0,3.0}, 
            } 
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("Truncator", new Truncator()); //this chops the uneven off.
        
        Pipeline model0 = new Pipeline();
        model0.add("Sine0", new Sine());

        Pipeline model1 = new Pipeline();
        model1.add("Cosine1", new Cosine());

        //if you're going to split and merge. 
        //unless you want to stack. don't put a classifier on the end of the pipelines.
        model.split("split", model0, model1);
        //TODO: fix for MultivariateSingleEnsemble
        //model.add("kNN", new MultivariateSingleEnsemble(new kNN()));

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }

    public static void example6() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0},
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0},
                {0.0,1.0,2.0,4.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 },
                {4.0,3.0}, 
                {0.0,1.0,2.0,4.0}
            } 
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[]{"A", "B"});

        // this is a sequential pipeline.
        Pipeline model = new Pipeline();
        model.add("Truncator", new Truncator()); //this chops the uneven off.
        
        Pipeline model0 = new Pipeline();
        model0.add("Sine0", new Sine());

        Pipeline model1 = new Pipeline();
        model1.add("Cosine1", new Cosine());

        //if you're going to split and merge. 
        //unless you want to stack. don't put a classifier on the end of the pipelines.


        //this will perform Pipeline one on dims 0 and 1
        //this will perform Pipeline two on dim 2
        model.split("split", new int[][]{{0,1},{2}}, model0, model1);
        //TODO: fix for MultivariateSingleEnsemble
        //model.add("kNN", new MultivariateSingleEnsemble(new kNN()));

        model.buildClassifier(data1);
        double[][] preds = model.distributionForInstances(data2);

        int count =0;
        int i=0;
        for(TimeSeriesInstance inst : data2){
            System.out.println(Arrays.toString(preds[i]));
            if(inst.getLabelIndex() == TimeSeriesSummaryStatistics.argmax(preds[i]))
                count++;

            i++;
        }

        System.out.println(count);

        double acc = (double) count / (double) data2.numInstances();
        System.out.println(acc);
    }

    

    public static void main(String[] args) throws Exception {
        example1();
        example2();
        example3();
        example4();
        example5();
        System.out.println("---------------------------- Example 6--------------------");
        example6();
    }
}
