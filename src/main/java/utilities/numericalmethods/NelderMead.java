/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.numericalmethods;

import java.io.*;
import java.util.*;
import java.text.DecimalFormat;
import java.util.function.Function;

/**
 *
 * @author a.bostrom1
 */
public class NelderMead {

    int NDIMS = 2;
    int NPTS = 3;
    final int MAXITER = 200;
    static int ncalls = 0;
    final double TOL = 1E-6;

    NumericalFunction func;

    
    
    double best_score;
    
    
    

    public static void main(String args[]) {
        NelderMead nm = new NelderMead();
        nm.descend(NelderMead::rosen, new double[]{1.3, 0.7, 0.8, 1.9, 1.2});
        System.out.println("\nThat's all folks");
    }
    private double[] best_result;
    
    
    public void descend(NumericalFunction fun, double[] x0) {
        
        func = fun;
        NDIMS = x0.length;
        NPTS = NDIMS+1;
        

        //n-dimensional simplex construction
        //based on matlab's fminsearch routine.
        double simplex[][] = new double[NPTS][NDIMS];
        
        simplex[0] = x0;
        
        for (int i = 1; i < NPTS; i++) {
            simplex[i] = new double[NDIMS];
            for (int j = 0; j < NDIMS; j++) {                
                simplex[i][j] = x0[j];
                
                //unit vector in j-th coordinate axis multiplied by hj;
                if(j==i){
                    simplex[i][j]+= x0[j] != 0 ? 0.05 : 0.00025;
                }
                   
            }
        } 
    
        descend(fun, simplex);
    }
    
    public void descend(NumericalFunction fun, double[][] simplex){
        
        func = fun;
        NDIMS = simplex[0].length;
        NPTS = NDIMS+1;
        double score[] = new double[NPTS];
        for (int i = 0; i < NPTS; i++) {
            score [i] = 0.0;
        }
        
        best_score = 1E99;

        //////////////// initialize the funcvals ////////////////
        for (int i = 0; i < NPTS; i++) {
            score[i] = func.FunctionToMinimise(simplex[i]);
        }

        System.out.println("ncalls = " + fwi(ncalls, 6));
        int iter = 0;

        for (iter = 1; iter < MAXITER; iter++) {
            /////////// identify lo, nhi, hi points //////////////

            double flo = score[0];
            double fhi = flo;
            int ilo = 0, ihi = 0, inhi = -1; // -1 means missing
            for (int i = 1; i < NPTS; i++) {
                if (score[i] < flo) {
                    flo = score[i];
                    ilo = i;
                }
                if (score[i] > fhi) {
                    fhi = score[i];
                    ihi = i;
                }
            }
            double fnhi = flo;
            inhi = ilo;
            for (int i = 0; i < NPTS; i++) {
                if ((i != ihi) && (score[i] > fnhi)) {
                    fnhi = score[i];
                    inhi = i;
                }
            }

            /*for (int j = 0; j < NDIMS; j++) {
                System.out.print(fwd(simplex[ilo][j], 18, 9));
            }
            System.out.print(fwd(score[ilo], 18, 9));
            
            System.out.println();*/

            ////////// exit criterion //////////////
            if ((iter % 4 * NDIMS) == 0) {
                if (score[ilo] > best_score - TOL) {
                    break;
                }
                best_score = score[ilo];
                best_result = simplex[ilo];
            }

            ///// compute ave[] vector excluding highest vertex //////
            double ave[] = new double[NDIMS];
            for (int j = 0; j < NDIMS; j++) {
                ave[j] = 0;
            }
            for (int i = 0; i < NPTS; i++) {
                if (i != ihi) {
                    for (int j = 0; j < NDIMS; j++) {
                        ave[j] += simplex[i][j];
                    }
                }
            }
            for (int j = 0; j < NDIMS; j++) {
                ave[j] /= (NPTS - 1);
            }

            ///////// try reflect ////////////////
            double r[] = new double[NDIMS];
            for (int j = 0; j < NDIMS; j++) {
                r[j] = 2 * ave[j] - simplex[ihi][j];
            }
            double fr = func.FunctionToMinimise(r);

            if ((flo <= fr) && (fr < fnhi)) // in zone: accept
            {
                System.arraycopy(r, 0, simplex[ihi], 0, NDIMS);
                score[ihi] = fr;
                continue;
            }

            if (fr < flo) //// below zone; try expand, else accept
            {
                double e[] = new double[NDIMS];
                for (int j = 0; j < NDIMS; j++) {
                    e[j] = 3 * ave[j] - 2 * simplex[ihi][j];
                }
                double fe = func.FunctionToMinimise(e);
                if (fe < fr) {
                    System.arraycopy(e, 0, simplex[ihi], 0, NDIMS);
                    score[ihi] = fe;
                    continue;
                } else {
                    System.arraycopy(r, 0, simplex[ihi], 0, NDIMS);
                    score[ihi] = fr;
                    continue;
                }
            }

            ///////////// above midzone, try contractions:
            if (fr < fhi) /// try outside contraction
            {
                double c[] = new double[NDIMS];
                for (int j = 0; j < NDIMS; j++) {
                    c[j] = 1.5 * ave[j] - 0.5 * simplex[ihi][j];
                }
                double fc = func.FunctionToMinimise(c);
                if (fc <= fr) {
                    System.arraycopy(c, 0, simplex[ihi], 0, NDIMS);
                    score[ihi] = fc;
                    continue;
                } else /////// contract
                {
                    for (int i = 0; i < NPTS; i++) {
                        if (i != ilo) {
                            for (int j = 0; j < NDIMS; j++) {
                                simplex[i][j] = 0.5 * simplex[ilo][j] + 0.5 * simplex[i][j];
                            }
                            score[i] = func.FunctionToMinimise(simplex[i]);
                        }
                    }
                    continue;
                }
            }

            if (fr >= fhi) /// over the top; try inside contraction
            {
                double cc[] = new double[NDIMS];
                for (int j = 0; j < NDIMS; j++) {
                    cc[j] = 0.5 * ave[j] + 0.5 * simplex[ihi][j];
                }
                double fcc = func.FunctionToMinimise(cc);
                if (fcc < fhi) {
                    System.arraycopy(cc, 0, simplex[ihi], 0, NDIMS);
                    score[ihi] = fcc;
                } 
                else ///////// contract
                {
                    for (int i = 0; i < NPTS; i++) {
                        if (i != ilo) {
                            for (int j = 0; j < NDIMS; j++) {
                                simplex[i][j] = 0.5 * simplex[ilo][j] + 0.5 * simplex[i][j];
                            }
                            score[i] = func.FunctionToMinimise(simplex[i]);
                        }
                    }
                }
            }
        }

        //System.out.println("ncalls, iters, Best =" + fwi(ncalls, 6) + fwi(iter, 6) + fwd(best_score, 16, 9));

    }
    
    public double getScore(){
        return best_score;
    }
    
    public double[] getResult(){
        return best_result;
    }
    
    

    static double func(double p[]) {
        ncalls++;
        return rosen(p);
    }

    static double rosen(double p[]) // Rosenbrock banana, two dimensions
    {
        double sum =0;
        for(int i=0; i< p.length/2; i++){
            double p1 = p[2*i];
            double p2 = p[2*i+1];
            
            double temp = ((p1*p1) - p2);
            double temp2 = temp*temp*100;
            
            double temp3 = p1 - 1;
            double temp4 = temp3*temp3;
            
            sum += temp2+ temp4;
        }
        
        /*double r0 = 10.0 * (p[1] - SQR(p[0]));
        double r1 = 1.0 - p[0];
        return SQR(r0) + SQR(r1);*/
        
        return sum;
    }

    static double parab(double p[]) // simple paraboloid
    {
        return SQR(p[0] - 2) + SQR(p[1] - 20);
    }

    /////////////////////////////////utilities ////////////////////
    static double SQR(double x) {
        return x * x;
    }

    static String fwi(int n, int w) // converts an int to a string with given width.
    {
        String s = Integer.toString(n);
        while (s.length() < w) {
            s = " " + s;
        }
        return s;
    }

    static String fwd(double x, int w, int d) // converts a double to a string with given width and decimals.
    {
        java.text.DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(d);
        df.setMinimumFractionDigits(d);
        df.setGroupingUsed(false);
        String s = df.format(x);
        while (s.length() < w) {
            s = " " + s;
        }
        if (s.length() > w) {
            s = "";
            for (int i = 0; i < w; i++) {
                s = s + "-";
            }
        }
        return s;
    }
}
