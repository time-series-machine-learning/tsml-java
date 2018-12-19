/*
This class is a helper class to describe the structure of our shapelet code and
* demonstrate how to use it.
 *copyright Anthony Bagnall
 * @author Anthony Bagnall, Jason Lines, Jon Hills and Edgaras Baranauskas
 */
package timeseriesweka.examples;

/* Package   weka.core.shapelet.* contains the classes 
 *          Shapelet that stores the actual shapelet, its location
 * in the data set, the quality assessment and a reference to the quality 
 * measure used
 *          BinaryShapelet that extends Shapelet to store the threshold used to 
 *  measure quality
 *          OrderLineObj: A simple class to store <distance,classValue> pairs 
 * for calculating the quality of a shapelet
 *          QualityMeasures: A class to store shapelet quality measure 
 * implementations. This includes an abstract quality measure class,
 * and implementations of each of the four shapelet quality measures
 *          QualityBound: A class to store shapelet quality measure bounding 
 * implementations. This is used to determine whether an early abandonment is 
 * permissible for the four quality measures.
 */
import timeseriesweka.filters.shapelet_transforms.ClusteredShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.distance_functions.ImprovedOnlineSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.CachedSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.distance_functions.OnlineSubSeqDistance;
import timeseriesweka.filters.shapelet_transforms.ApproximateShapeletTransform;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;


/* package weka.filters.timeseries.shapelet_transforms.* contains
 *      FullShapeletTransform: Enumerative search to find the best k shapelets.
 *        ShapeletTransformDistCaching: subclass of FullShapeletTransform that 
 * uses the distance caching algorithm described in Mueen11. This is the fastest
 * exact approach, but is memory intensive. 
 *        ShapeletTransform: subclass of FullShapeletTransform that uses  
 distance online normalisation and early abandon described in ??. Not as fast,
 * but does not require the extra memory.
 *      ClusteredShapeletTransform: contains a FullShapeletTransform, and does post 
 * transformation clustering. 
*       
* */

/* package weka.classifiers.trees.shapelet_trees.* contains
 *  ShapeletTreeClassifier: implementation of a shapelet tree to match the 
 * description on the original paper.
 * 4x tree classifiers based on the alternative distance measures in class 
 * QualityMeasures.
 */
import weka.core.*;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality.ShapeletQualityChoice;
public class ShapeletExamples {

    public static ShapeletTransform st;
    public static Instances basicTransformExample(Instances train){
 /*Class to demonstrate the usage of the ShapeletTransform. Returns the 
  * transformed set of instances  
  */
        st =new ShapeletTransform();
        st.setSubSeqDistance(new OnlineSubSeqDistance());
/*The number of shapelets defaults to 100. we recommend setting it to a large
value, since there will be many duplicates and there is little overhead in 
* keeping a lot (although the shapelet early abandon becomes less efficient).
* 
*/
//Let m=train.numAttributes()-1 (series length)
//Let n=   train.numInstances() (number of series)      
        int nosShapelets=(train.numAttributes()-1)*train.numInstances()/5;
        if(nosShapelets<ShapeletTransform.DEFAULT_NUMSHAPELETS)
            nosShapelets=ShapeletTransform.DEFAULT_NUMSHAPELETS;
        st.setNumberOfShapelets(nosShapelets);
/* Two other key parameters are minShapeletLength and maxShapeletLength. For 
 * each value between these two, a full search is performed, which is 
 * order (m^2n^2), so clearly there is a time/accuracy trade off. Defaults 
 * to min of 3 max of 30.
 */
        int minLength=5;
        int maxLength=(train.numAttributes()-1)/10;
        if(maxLength<ShapeletTransform.DEFAULT_MINSHAPELETLENGTH)
            maxLength=ShapeletTransform.DEFAULT_MINSHAPELETLENGTH;
        st.setShapeletMinAndMax(minLength, maxLength);

/*Next you need to set the quality measure. This defaults to IG, but         
 * we recommend using the F stat. It is faster and (debatably) more accurate.
 */
        st.setQualityMeasure(ShapeletQualityChoice.F_STAT);
// You can set the filter to output details of the shapelets or not  
        st.setLogOutputFile("ShapeletExampleLog.csv");
// Alternatively, you can turn the logging off
//        st.turnOffLog();        
 
/* Thats the basic options. Now you need to perform the transform.
 * FullShapeletTransform extends the weka SimpleBatchFilter, but we have made 
 * the method process public to make usage easier.
 */
        Instances shapeletT=null;
        try {
            shapeletT=st.process(train);
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }
        return shapeletT;
    }
    
    public static Instances clusteredShapeletTransformExample(Instances train){
/* The class ClusteredShapeletTransform contains a FullShapeletTransform and
 * post transform clusters it. You can either perform the transform outside of 
 * the ClusteredShapeletTransform or leave it to do it internally.
 * 
 */

        Instances shapeletT=null;
//Cluster down to 10% of the number.        
        int nosShapelets=(train.numAttributes()-1)*train.numInstances()/50;
        ClusteredShapeletTransform cst = new ClusteredShapeletTransform(st,nosShapelets);
        System.out.println(" Clustering down to "+nosShapelets+" Shapelets");
        System.out.println(" From "+st.getNumberOfShapelets()+" Shapelets");
        
        try {
            shapeletT=cst.process(train);
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet clustering"+ex);
            
            ex.printStackTrace();
            System.exit(0);
        }
        return shapeletT;

    }
    
    public static void initializeShapelet(ShapeletTransform s,Instances train){
//       int nosShapelets=(train.numAttributes()-1)*train.numInstances()/5;
       s.setNumberOfShapelets(1);        
       int minLength=15;
       int maxLength=36;
//       int maxLength=(train.numAttributes()-1)/10;
       s.setShapeletMinAndMax(minLength, maxLength);
       s.setQualityMeasure(ShapeletQualityChoice.F_STAT);
       s.supressOutput();
       s.turnOffLog();
    }
    public static void distanceOptimizations(Instances train){
        Instances shapeletT=null;

        SubSeqDistance[] ssq = {new SubSeqDistance(), new OnlineSubSeqDistance(), new CachedSubSeqDistance(), new ImprovedOnlineSubSeqDistance()};
        ShapeletTransform[] st = new ShapeletTransform[ssq.length];
        for(int i=0; i< st.length; i++){
            st[i] = new ShapeletTransform();
            st[i].setSubSeqDistance(ssq[i]);
            initializeShapelet(st[i], train);
        }   

        DecimalFormat df =new DecimalFormat("###.####");
        long t1=0;
        long t2=0;
        double[] time = new double[ssq.length];
        try {
            
            for(int i=0; i< time.length; i++){
                t1=System.currentTimeMillis();
                shapeletT=st[i].process(train);
                t2=System.currentTimeMillis();
                time[i] =((t2-t1)/1000.0);
            }
            
            System.out.println("TIME (seconds)");
            for(int i=0; i< time.length; i++){
                System.out.print(ssq[i].getClass().getSimpleName()+ "\t");
            }
            System.out.println();
            for(int i=0; i< time.length; i++){
                System.out.print(df.format(time[i])+ "\t");
            }
            System.out.println();
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }       
    }
    public static void shapeletEarlyAbandons(Instances train){
//Time the speed up from early abandon of the four distance measures.

        //IG:         
        ShapeletTransform[] s=new ShapeletTransform[4];
        ShapeletTransform[] pruned=new ShapeletTransform[4];
        for(int i=0;i<s.length;i++){
            s[i]=new ShapeletTransform();
            pruned[i]=new ShapeletTransform();
        }
        for(ShapeletTransform s1:s){
            initializeShapelet(s1,train);
            s1.setCandidatePruning(false);
        }
        for(ShapeletTransform s1:pruned){
            initializeShapelet(s1,train);
            s1.setCandidatePruning(true);
        }
        ShapeletQualityChoice[] choices=ShapeletQualityChoice.values();
        for(int i=0;i<s.length;i++){
            s[i].setQualityMeasure(choices[i]);
            pruned[i].setQualityMeasure(choices[i]);
        }
        long t1,t2;
        double time1,time2;
        DecimalFormat df =new DecimalFormat("###.####");
        try {
            for(int i=0;i<s.length;i++){
                t1=System.currentTimeMillis();
                s[i].process(train);
                t2=System.currentTimeMillis();
                time1=((t2-t1)/1000.0);
                t1=System.currentTimeMillis();
                pruned[i].process(train);
                t2=System.currentTimeMillis();
                time2=((t2-t1)/1000.0);
                System.out.println(" ********* QUALITY MEASURE ="+s[i].getQualityMeasure()+"  **********");
                System.out.println(" NO ABANDON \t\t ABANDON\t\t ABANDON/(NO ABANDON)%\t\t SPEED UP ");
                System.out.println(df.format(time1)+"\t\t\t"+df.format(time2)+"\t\t\t"+(int)(100.0*(time2/time1))+"%"+"\t\t\t"+df.format(time1/time2));
            }
       } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }       
        
    }

    public static Instances approxDataTransformExample(Instances train){
        /*Class to demonstrate the usage of the ApproximateShapeletTransform. Returns the 
         * transformed set of instances  
         */
        st = new ApproximateShapeletTransform();
        
        //Parameters that are relevant to all types of transforms that extend FullShapeletTransform:
        //1. Number of shapelets to be stored
        int nosShapelets=(train.numAttributes()-1)*train.numInstances()/5;
        if(nosShapelets<ShapeletTransform.DEFAULT_NUMSHAPELETS)
            nosShapelets=ShapeletTransform.DEFAULT_NUMSHAPELETS;
        st.setNumberOfShapelets(nosShapelets);
        //2. Shapelet lenght range to be eplored
        int minLength=5;
        int maxLength=(train.numAttributes()-1)/10;
        if(maxLength<ShapeletTransform.DEFAULT_MINSHAPELETLENGTH)
            maxLength=ShapeletTransform.DEFAULT_MINSHAPELETLENGTH;
        st.setShapeletMinAndMax(minLength, maxLength);
        //3. Quality measure
        st.setQualityMeasure(ShapeletQualityChoice.F_STAT);
        
        //4. Set the filter to output details of the shapelets or not  
        st.setLogOutputFile("ApproximateTransformExampleLog.csv");
        
        /* Parameters that are specific to ApproximateShapeletTransform are:
         * 1. Dataset sampling level - specifies the percentage of instances to be used
         *    from the provided training data for the shapelet discovery, i.e. setting
         *    this parmeter to 50 forces the transform to sample the training data to
         *    reduce it to 50% of the original size.
         *
         * 2. Series reduction level - specifies the percentage of how much each 
         *    series should be reduced, i.e. setting this parameter to 50 forces 
         *    the trasform to approximate each series using PAA such that each series
         *    lenght is 50% of the original length. 
         *    
         * On average the higher the percentage the lower the accuracy is to be 
         * expected. For example setting the levels to 50 - 50  on averege 
         * should reduce the processing time by ~30 times and reduce the accuracy by
         * ~15%
         */ 
            
        try {
            // Parameter 1 - datast sampling level, Parameter 2 - PAA approximation level
            ((ApproximateShapeletTransform)st).setSampleLevels(50, 50);
        } catch (IOException ex) {
            Logger.getLogger(ShapeletExamples.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        // Now perform the transform exacty like using the ShapeletTransfomr.
        Instances shapeletT=null;
        try {
            shapeletT=st.process(train);
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }
        return shapeletT;
    }
	
    public static void main(String[] args){
        Instances train=null,test=null;
        FileReader r;
        
        String path = "C:\\LocalData\\Dropbox\\TSC Problems\\";
        String dataset = "SonyAIBORobotSurface1";
        try{		
            r= new FileReader(path+dataset+"\\"+dataset+"_TRAIN.arff"); 
            train = new Instances(r); 
            train.setClassIndex(train.numAttributes()-1);
            r= new FileReader(path+dataset+"\\"+dataset+ "_TEST.arff"); 
            test = new Instances(r); 
            test.setClassIndex(test.numAttributes()-1);

        }
        catch(Exception e)
        {
                System.out.println("Unable to load data. Exception thrown ="+e);
                System.exit(0);
        }
        System.out.println("****************** PERFORMING BASIC TRANSFORM *******");
        Instances shapeletT=basicTransformExample(train);
        System.out.println(" Transformed data set ="+shapeletT);
        System.out.println("\n **************** CLUSTERING *******");
        shapeletT=clusteredShapeletTransformExample(train);
        System.out.println(" Clustered Transformed data set ="+shapeletT);
        System.out.println("\n ******Distance calculation optimizations *******");
        distanceOptimizations(train);                
        System.out.println("\n ******Shapelet Early Abandons *******");
        shapeletEarlyAbandons(train);               
    }
}
