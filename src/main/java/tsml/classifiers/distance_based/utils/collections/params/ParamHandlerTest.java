package tsml.classifiers.distance_based.utils.collections.params;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace.newDDTWDistance;
import static tsml.classifiers.distance_based.distances.wdtw.spaces.WDDTWDistanceSpace.newWDDTWDistance;

@RunWith(Parameterized.class)
public class ParamHandlerTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                { new DTWDistance() },
                { new ERPDistance() },
                { new LCSSDistance() },
                { new MSMDistance() },
                { new EDistance() },
                { new WDTWDistance() },
                { new TWEDistance() },
                { newWDDTWDistance() },
                { newDDTWDistance() },
        });
    }
    
    @Parameterized.Parameter(0)
    public Object handler;
    
    public Object getHandler() {
        return CopierUtils.deepCopy(handler);
    }
    
//    @Test()
//    public void testSetMissingParams() {
//        final Object handler = getHandler();
//        try {
//            ParamHandlerUtils.setParams(handler, new ParamSet().add("this is the missing flag", 0.6));
//            Assert.fail("expected exception on invalid parameter");
//        } catch(RuntimeException ignored) {}
//    }

    @Test
    public void testGetParams() {
        final Object handler = getHandler();
        final ParamSet params = ParamHandlerUtils.getParams(handler);
    }
    
    @Test
    public void testSetParams() throws Exception {
        final Object handler = getHandler();
        final ParamSet paramSet = ParamHandlerUtils.getParams(handler);
        
        for(String name : paramSet.keySet()) {
            final Object value = paramSet.get(name);
            if(
                    value instanceof Double || 
                    value instanceof Float || 
                    value instanceof Integer || 
                    value instanceof Byte || 
                    value instanceof Short || 
                    value instanceof Character || 
                    value instanceof Long
            ) {
                ParamHandlerUtils.setParams(handler, new ParamSet().add(name, "0"));
                Assert.assertEquals(0d, (double) ParamHandlerUtils.getParams(handler).get(name), 0d);
                ParamHandlerUtils.setParams(handler, new ParamSet().add(name, "1"));
                Assert.assertEquals(1d, (double) ParamHandlerUtils.getParams(handler).get(name), 0d);
            } else {
                // non primitive type
                // get the type, make fresh copy of it. This will then be != to the previous value
                final Object copyA = CopierUtils.deepCopy(value);
                final Object copyB = CopierUtils.deepCopy(value);
                ParamHandlerUtils.setParams(handler, new ParamSet().add(name, copyA));
                Assert.assertNotSame(value, ParamHandlerUtils.getParams(handler).get(name));
                ParamHandlerUtils.setParams(handler, new ParamSet().add(name, copyB));
                Assert.assertNotSame(copyA, ParamHandlerUtils.getParams(handler).get(name));
            }
        }
    }
}
