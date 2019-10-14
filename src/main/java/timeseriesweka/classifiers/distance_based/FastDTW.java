/*

Wrapper for a DTW implementation that speeds up the window size search
through caching

 */
package timeseriesweka.classifiers.distance_based;


import java.util.ArrayList;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.EnhancedAbstractClassifier;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.FastWWSByPercent;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.WindowSearcher;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class FastDTW extends EnhancedAbstractClassifier{

    WindowSearcher ws;
    protected ArrayList<Double> buildTimes;
    
    public FastDTW(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        ws=new FastWWSByPercent();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        long startTime=System.nanoTime(); 
        ws.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime()-startTime);
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        trainResults.setMemory(usedBytes);
    }
    public double classifyInstance(Instance data) throws Exception {
       return ws.classifyInstance(data);
    }

    @Override
    public String getParameters() {
        String result="CVAcc,"+trainResults.getAcc()+",Memory,"+trainResults.getMemory();
        result+=",WindowSize,"+ws.getBestWin()+",Score,"+ws.getBestScore();
        return result;
    }

    
}
