/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import multivariate_timeseriesweka.classifiers.ConcatenateClassifier;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.ShapeletTransformClassifier;

/**
 *
 * @author raj09hxu
 */
public class ConcatenationClassifiers {
    
    public static final Map<String, Supplier<ConcatenateClassifier>> CONCAT_CLASSIFIERS;
    static {
        Map<String, Supplier<ConcatenateClassifier>> map = new HashMap();
        map.put("ST_HESCA", ConcatenationClassifiers::createST_HESCA_concat);
        map.put("BOSS", ConcatenationClassifiers::createBOSS_concat);
        map.put("EE",ConcatenationClassifiers::createEE_concat);
        map.put("LS",ConcatenationClassifiers::createLS_concat);
        CONCAT_CLASSIFIERS = Collections.unmodifiableMap(map);
    }
    
    
    public static ConcatenateClassifier createST_HESCA_concat(){
        ShapeletTransformClassifier st = new ShapeletTransformClassifier();
        st.setDayLimit(1);
        return new ConcatenateClassifier(st);
    }   
    
    public static ConcatenateClassifier createBOSS_concat(){
        return new ConcatenateClassifier(new BOSS());
    }
    
    public static ConcatenateClassifier createEE_concat(){
        return new ConcatenateClassifier(new ElasticEnsemble());
    }
    
    public static ConcatenateClassifier createLS_concat(){
        return new ConcatenateClassifier(new LearnShapelets());
       
    }
    
}
