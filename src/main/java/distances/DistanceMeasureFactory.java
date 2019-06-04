package distances;

import distances.ddtw.Ddtw;
import distances.dtw.Dtw;
import distances.erp.Erp;
import distances.lcss.Lcss;
import distances.msm.Msm;
import distances.twe.Twe;
import distances.wddtw.Wddtw;
import distances.wdtw.Wdtw;

public class DistanceMeasureFactory {
    public static DistanceMeasure fromString(String str) {
        str = str.toLowerCase();
        switch(str) {
            case "dtw": return new Dtw();
            case "ddtw": return new Ddtw();
            case "wdtw": return new Wdtw();
            case "wddtw": return new Wddtw();
            case "twe": return new Twe();
            case "msm": return new Msm();
            case "lcss": return new Lcss();
            case "erp": return new Erp();
            default: throw new IllegalArgumentException("unknown distance measure: " + str);
        }
    }
}
