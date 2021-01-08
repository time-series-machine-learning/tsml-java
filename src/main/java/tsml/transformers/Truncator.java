package tsml.transformers;

import experiments.data.DatasetLists;
import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.stream.IntStream;

/**
 * Class to truncate series to make them all equal length. In Weka unequal length series are padded with missing values
 * to avoid ragged arrays. This class removes observations from the series to make them all as short as the shortest in
 * the train data This may have involved some prior preprocessing.
 * there is an edge case where the shortest in the test data is shorter than the shortest in the Train data. In this scenario
 * the solution is to pad the particular test series with this characteristic.
 * Assumptions:
 *  1. Data loaded from ARFF format are already padded to be the same length for Train and Test.
 *  2. All missing values at the end of a series are padding
 *
 *  todo: implement univariate
 *  todo: implement multivariate
 *  todo: test inc edge cases
 *  todo: decide on whether to clone or not
 * @author Tony Bagnall 1/1/2020
 */
public class Truncator implements TrainableTransformer{

    int shortestSeriesLength=Integer.MAX_VALUE;
    private boolean isFit;

    /**
     * Determine the length of the shortest series in the data
     * store internally in shortestSeriesLength
     * @param data: either univariate or multivariate time series classification problem
     */
    @Override
    public void fit(Instances data) {
        if(data.attribute(0).isRelationValued()) {    //Multivariate
            for(Instance ins:data){
                Instances d=ins.relationalValue(0);
                for(Instance internal:d){
                    int l=findLength(internal,false);
                    if (l < shortestSeriesLength)
                        shortestSeriesLength = l;
                }
            }
        }
        else{
            for(Instance ins:data) {
                int l = findLength(ins, true);
                if (l < shortestSeriesLength)
                    shortestSeriesLength = l;
            }
        }

        isFit = true;
    }

    /**
     * Shortens
     * @param data
     * @return
     */
    @Override
    public Instances transform(Instances data) {
        if(data.attribute(0).isRelationValued()) {    //Multivariate
             for(Instance ins:data){
                 while (ins.numAttributes() > shortestSeriesLength)
                     ins.deleteAttributeAt(ins.numAttributes() );
             }
        }
        else {
            while (data.numAttributes() - 2 >= shortestSeriesLength)
                data.deleteAttributeAt(data.numAttributes() - 2);
        }
        return data;
    }

    @Override
    public Instance transform(Instance ins) {
        if(ins.attribute(0).isRelationValued()) {    //Multivariate
            while (ins.numAttributes() > shortestSeriesLength)
                ins.deleteAttributeAt(ins.numAttributes() );
        }
       else {
            while (ins.numAttributes() - 1 > shortestSeriesLength)
                ins.deleteAttributeAt(ins.numAttributes() - 1);
        }
        return ins;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        throw new IllegalArgumentException(" determine Output Format is not implemented for Truncator, there is no need");
    }

    public static void testTruncateUnivariateData(String problemPath) {
        Instances train = DatasetLoading.loadData(problemPath+"AllGestureWiimoteX/AllGestureWiimoteX_TRAIN");
        Instances test = DatasetLoading.loadData(problemPath+"AllGestureWiimoteX/AllGestureWiimoteX_TEST");
        System.out.println(" Test on unequal length series AllGestureWiimoteX: Min series length should be 11 in train and 2 in test ");
        Truncator t = new Truncator();
        t.fit(train);
        train=t.transform(train);
        test=t.transform(test);
        System.out.print("Num atts ="+train.numAttributes()+"\n Train instance 1 = ");
        double[] d= train.instance(0).toDoubleArray();
        for(double x:d)
            System.out.print(x+",");
        System.out.print("\n");
        String[] probs=DatasetLists.variableLengthUnivariate;
        System.out.print("Num atts ="+(test.numAttributes()-1)+"\n Test instance 1 = ");
        double[] d2= test.instance(0).toDoubleArray();
        for(double x:d2)
            System.out.print(x+",");
        System.out.print("\n");
//        String[] probs=DatasetLists.variableLengthUnivariate;

    }
    /**
     * find the length of the series ins by counting missing values from the end
     * @param ins
     * @param classValuePresent
     * @return
     */
    public static int findLength(Instance ins, boolean classValuePresent){
        int seriesLength = classValuePresent?ins.numAttributes()-2:ins.numAttributes()-1;
        while(seriesLength>0 && ins.isMissing(seriesLength))
            seriesLength--;
        seriesLength++;//Correct for zero index
        return seriesLength;

    }


    public static int[] univariateSeriesLengths(String problemPath){
        String[] probs=DatasetLists.variableLengthUnivariate;
        System.out.println(" Total number of problesm = "+probs.length);
        int[][] minMax=new int[probs.length][4];
        //        String[] probs=DatasetLists.variableLength2018Problems;
//        String[] probs=new String[]{"AllGestureWiimoteX"};
//        OutFile of=new OutFile(problemPath+"Unqu");
        int count=0;
        for(String str:probs){
            Instances train = DatasetLoading.loadData(problemPath+str+"/"+str+"_TRAIN");
            Instances test = DatasetLoading.loadData(problemPath+str+"/"+str+"_TEST");
            int min=Integer.MAX_VALUE;
            int max=0;
            for(Instance ins:train){
                int length=findLength(ins,true);
                if(length>max)
                    max=length;
                if(length<min)
                    min=length;
//                System.out.println("Length ="+length+" last value = "+ins.value(length-1)+" one after = "+ins.value(length));
            }
            if(min!=max)
                System.out.print(" PROBLEM "+str+"\t\tTRAIN MIN ="+min+"\t Max ="+max);
            minMax[count][0]=min;
            minMax[count][1]=max;

            min=Integer.MAX_VALUE;
            max=0;
            for(Instance ins:test){
                int length=findLength(ins,true);
                if(length>max)
                    max=length;
                if(length<min)
                    min=length;
//                System.out.println("Length ="+length+" last value = "+ins.value(length-1)+" one after = "+ins.value(length));
            }
            if(min!=max)
                System.out.println("\t"+str+"\t TEST MIN ="+min+"\t Max ="+max);
            minMax[count][2]=min;
            minMax[count][3]=max;
            count++;
        }
        for(int i=0;i<probs.length;i++){
            System.out.println("{"+minMax[i][0]+","+minMax[i][1]+","+minMax[i][2]+","+minMax[i][3]+"},");
        }
        return null;
    }



    public static int[] multivariateSeriesLengths(String problemPath){
        String[] probs=DatasetLists.variableLengthMultivariate;
        ArrayList<String> names=new ArrayList<>();
        int[][] minMax=new int[probs.length][4];
        System.out.println(" Total number of problems = "+probs.length);
        //        String[] probs=DatasetLists.variableLength2018Problems;
//        String[] probs=new String[]{"AllGestureWiimoteX"};
//        OutFile of=new OutFile(problemPath+"Unqu");
        int count=0;
        for(String str:probs){
//            System.out.print(" PROBLEM "+str);
            Instances train = DatasetLoading.loadData(problemPath+str+"/"+str+"_TRAIN");
            Instances test = DatasetLoading.loadData(problemPath+str+"/"+str+"_TEST");
            int min=Integer.MAX_VALUE;
            int max=0;
            for(Instance ins:train){
                Instances cs=ins.relationalValue(0);
                for(Instance in:cs) {
                    int length = findLength(in, true);
                    if (length > max)
                        max = length;
                    if (length < min)
                        min = length;
                }
                    //                System.out.println("Length ="+length+" last value = "+ins.value(length-1)+" one after = "+ins.value(length));
            }
            if(min!=max) {
                System.out.print("\t\tTRAIN MIN =" + min + "\t Max =" + max);
                names.add(str);
            }
            minMax[count][0]=min;
            minMax[count][1]=max;
            min=Integer.MAX_VALUE;
            max=0;
            for(Instance ins:test){
                Instances cs=ins.relationalValue(0);
                for(Instance in:cs) {
                    int length = findLength(in, true);
                    if (length > max)
                        max = length;
                    if (length < min)
                        min = length;
                }
            }
            minMax[count][2]=min;
            minMax[count][3]=max;
            if(min!=max)
                System.out.println("\t"+str+"\t TEST MIN ="+min+"\t Max ="+max);
            count++;
        }
        for(String str:names)
            System.out.print("\""+str+"\",");
        for(int i=0;i<probs.length;i++){
            System.out.println("{"+minMax[i][0]+","+minMax[i][1]+","+minMax[i][2]+","+minMax[i][3]+"},");
        }

        return null;
    }



    public static void main(String[] args) {

        String path="Z:\\ArchiveData\\Univariate_arff\\";
        String path2="Z:\\ArchiveData\\Multivariate_arff\\";

//        String path="Z:\\ArchiveData\\Univariate_arff\\";
        multivariateSeriesLengths(path2);
 //       univariateSeriesLengths(path);
     System.exit(0);
        System.out.println(" Test 1, univariate data with padding");
        testTruncateUnivariateData(path);
        System.exit(0);
        System.out.println(" Test 2, univariate data with no padding");
        System.out.println(" Test 3, multivariate data with padding");
        testTruncateUnivariateData("");
        System.out.println(" Test 2, multivariate data with no padding");


    }

    @Override
    public boolean isFit() {
        return isFit;
    }


    
	@Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        return inst.getVSlice(IntStream.range(0, shortestSeriesLength).toArray());
    }
    
	@Override
	public void fit(TimeSeriesInstances data) {
		shortestSeriesLength = data.getMinLength();
        isFit = true;
	}

}
