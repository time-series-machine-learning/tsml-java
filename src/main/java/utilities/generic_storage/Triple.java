/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.generic_storage;

/**
 *
 * @author raj09hxu
 */
public class Triple <T1, T2, T3> extends Pair<T1, T2>{
    public final T3 var3;
    public Triple(T1 t1, T2 t2, T3 t3){
        super(t1,t2);
        var3 = t3;
    }
    
    @Override
    public String toString(){
        return super.toString() + " " + var3;
    }
}
