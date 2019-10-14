/*

Wrapper for a DTW implementation that speeds up the window size search
through caching

 */
package timeseriesweka.classifiers.distance_based;


import java.util.ArrayList;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.FastWWSByPercent;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.WindowSearcher;
import weka.classifiers.AbstractClassifier;
import timeseriesweka.classifiers.AbstractClassifierWithTrainingInfo;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class FastDTW extends AbstractClassifierWithTrainingInfo{

    WindowSearcher ws;
    protected ArrayList<Double> buildTimes;
    protected ClassifierResults res =new ClassifierResults();
    
    public FastDTW(){
        ws=new FastWWSByPercent();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        long startTime=System.nanoTime(); 
        ws.buildClassifier(data);
        res.setBuildTime(System.nanoTime()-startTime);
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        res.setMemory(usedBytes);
    }
    public double classifyInstance(Instance data) throws Exception {
       return ws.classifyInstance(data);
    }

    @Override
    public String getParameters() {
        String result="CVAcc,"+res.getAcc()+",Memory,"+res.getMemory();
        result+=",WindowSize,"+ws.getBestWin()+",Score,"+ws.getBestScore();
        return result;
    }

    
}
