package distances.twe;

import distances.DistanceMeasure;

public class Twe extends DistanceMeasure {

    @Override
    protected double measureDistance(double[] a, double[] b, double cutOff) {
        /*This code is faithful to the c version, so uses a redundant
 * Multidimensional representation. The c code does not describe what the
            arguments
 * tsB and tsA are. We assume they are the time stamps (i.e. index sets),
 * and initialise them accordingly.
 */

        // todo cleanup
        // todo trim memory to window by window
        // todo early abandon
        // todo might be able to inherit from dtw to use warping window perhaps?

        int dim=1;
        double dist, disti1, distj1;
        double[][] ta=new double[a.length][dim];
        double[][] tb=new double[a.length][dim];
        double[] tsa=new double[a.length];
        double[] tsb=new double[b.length];
        for(int i=0;i<tsa.length;i++)
            tsa[i]=(i+1);
        for(int i=0;i<tsb.length;i++)
            tsb[i]=(i+1);

        int r = ta.length;
        int c = tb.length;
        int i,j,k;
//Copy over values
        for(i=0;i<a.length;i++)
            ta[i][0]=a[i];
        for(i=0;i<b.length;i++)
            tb[i][0]=b[i];

        /* allocations in c
	double **D = (double **)calloc(r+1, sizeof(double*));
	double *Di1 = (double *)calloc(r+1, sizeof(double));
	double *Dj1 = (double *)calloc(c+1, sizeof(double));
	for(i=0; i<=r; i++) {
		D[i]=(double *)calloc(c+1, sizeof(double));
	}
*/
        double [][]D = new double[r+1][c+1];
        double[] Di1 = new double[r+1];
        double[] Dj1 = new double[c+1];
// local costs initializations
        for(j=1; j<=c; j++) {
            distj1=0;
            for(k=0; k<dim; k++)
                if(j>1){
//CHANGE AJB 8/1/16: Only use power of 2 for speed up,
                    distj1+=(tb[j-2][k]-tb[j-1][k])*(tb[j-2][k]-tb[j-1][k]);
// OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
// in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                }
                else
                    distj1+=tb[j-1][k]*tb[j-1][k];
//OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
            Dj1[j]=(distj1);
        }

        for(i=1; i<=r; i++) {
            disti1=0;
            for(k=0; k<dim; k++)
                if(i>1)
                    disti1+=(ta[i-2][k]-ta[i-1][k])*(ta[i-2][k]-ta[i-1][k]);
// OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else
                    disti1+=(ta[i-1][k])*(ta[i-1][k]);
//OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i]=(disti1);

            for(j=1; j<=c; j++) {
                dist=0;
                for(k=0; k<dim; k++){
                    dist+=(ta[i-1][k]-tb[j-1][k])*(ta[i-1][k]-tb[j-1][k]);
//                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if(i>1&&j>1)
                        dist+=(ta[i-2][k]-tb[j-2][k])*(ta[i-2][k]-tb[j-2][k]);
//                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j]=(dist);
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0]=0;
        for(i=1; i<=r; i++)
            D[i][0]=D[i-1][0]+Di1[i];
        for(j=1; j<=c; j++)
            D[0][j]=D[0][j-1]+Dj1[j];

        double dmin, htrans, dist0;
        int iback;

        for (i=1; i<=r; i++){
            for (j=1; j<=c; j++){
                htrans=Math.abs((tsa[i-1]-tsb[j-1]));
                if(j>1&&i>1)
                    htrans+=Math.abs((tsa[i-2]-tsb[j-2]));
                dist0=D[i-1][j-1]+ stiffness *htrans+D[i][j];
                dmin=dist0;
                if(i>1)
                    htrans=((tsa[i-1]-tsa[i-2]));
                else htrans=tsa[i-1];
                dist=Di1[i]+D[i-1][j]+ penalty + stiffness *htrans;
                if(dmin>dist){
                    dmin=dist;
                }
                if(j>1)
                    htrans=(tsb[j-1]-tsb[j-2]);
                else htrans=tsb[j-1];
                dist=Dj1[j]+D[i][j-1]+ penalty + stiffness *htrans;
                if(dmin>dist){
                    dmin=dist;
                }
                D[i][j] = dmin;
            }
        }

        dist = D[r][c];
        return dist;
    }

    public Twe(double penalty, double stiffness) {
        setPenalty(penalty);
        setStiffness(stiffness);
    }

    public Twe() {
        this(1,1);
    }

    private double penalty; // between 0 and 1 inclusively, the penalty for deletions AKA lambda

    public double getPenalty() {
        return penalty;
    }

    public void setPenalty(double penalty) {
        this.penalty = penalty;
    }

    public double getStiffness() {
        return stiffness;
    }

    public void setStiffness(double stiffness) {
        this.stiffness = stiffness;
    }

    private double stiffness; // above 0, nu or 'stiffness'. 1e10^-5, 1e10^-4, 1e10^-3, 1e10^-2, 1e10^-1, 1

    public static final String STIFFNESS_KEY = "stiffness";
    public static final String PENALTY_KEY = "penalty";

    public static void main(String[] args) {
        double[] a = new double[] {1,2,3,4,5,6,7,8,9,10};
        double[] b = new double[] {10,9,8,7,6,5,4,3,2,1};
        Twe twe = new Twe();
        double distance = twe.measureDistance(a, b, Double.POSITIVE_INFINITY);
        System.out.println(distance);
    }

    @Override
    public void setOptions(String[] options) {
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if(key.equals(STIFFNESS_KEY)) {
                setStiffness(Double.parseDouble(value));
            } else if(key.equals(PENALTY_KEY)) {
                setPenalty(Double.parseDouble(value));
            }
        }
    }

    @Override
    public String[] getOptions() {
        return new String[] {
            STIFFNESS_KEY,
            String.valueOf(stiffness),
            PENALTY_KEY,
            String.valueOf(penalty)
        };
    }
}
