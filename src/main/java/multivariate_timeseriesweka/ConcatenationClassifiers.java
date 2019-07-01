/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
