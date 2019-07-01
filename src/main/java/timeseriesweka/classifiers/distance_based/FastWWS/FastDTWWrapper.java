/*Wrapper for Francois Petitjean's fast DTW code

 */
package timeseriesweka.classifiers.distance_based.FastWWS;


import java.util.ArrayList;
import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.SaveParameterInfo;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.FastWWSByPercent;
import timeseriesweka.classifiers.distance_based.FastWWS.windowSearcher.WindowSearcher;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class FastDTWWrapper extends AbstractClassifier  implements SaveParameterInfo{

    WindowSearcher ws;
    protected ArrayList<Double> buildTimes;
    protected ClassifierResults res =new ClassifierResults();
    
    public FastDTWWrapper(){
        ws=new FastWWSByPercent();
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        long startTime=System.currentTimeMillis(); 
        ws.buildClassifier(data);
        res.setBuildTime(System.currentTimeMillis()-startTime);
        Runtime rt = Runtime.getRuntime();
        long usedBytes = (rt.totalMemory() - rt.freeMemory());
        res.setMemory(usedBytes);
    }
    public double classifyInstance(Instance data) throws Exception {
       return ws.classifyInstance(data);
       
    }

    @Override
    public String getParameters() {
        String result="BuildTime,"+res.getBuildTime()+",CVAcc,"+res.getAcc()+",Memory,"+res.getMemory();
        result+=",WindowSize,"+ws.getBestWin()+",Score,"+ws.getBestScore();
        return result;
    }

    
}
