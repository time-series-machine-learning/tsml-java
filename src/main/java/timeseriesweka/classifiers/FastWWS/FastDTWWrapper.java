/*Wrapper for Francois Petitjean's fast DTW code

 */
package timeseriesweka.classifiers.FastWWS;


import java.util.ArrayList;
import timeseriesweka.classifiers.FastWWS.windowSearcher.*;
import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.SaveParameterInfo;
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
