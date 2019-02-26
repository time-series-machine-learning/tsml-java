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
package utilities;

/**
 *
 * @author xmw13bzu
 */
public class Timer {
    
    public static boolean PRINT = false;
    
    public static double SECS = 1000.0;
    
    long startTime;
    String name;
    
    public Timer(String name) {
        this(name, true);
    }
    
    public Timer() {
        this("", true);
    }
    
    public Timer(String name, boolean auto_start){
        this.name = name;
        if(auto_start)
            start();
    }
    
    public void start() {
        startTime = System.currentTimeMillis();
    }
    
    public long restart() { 
        long t = timeSoFar();
        startTime = System.currentTimeMillis();
        return t;
    }
    
    public long timeSoFar() {
        return System.currentTimeMillis() - startTime;
    }
    
    @Override
    public String toString() {
        return "("+name+") TIMER timeSoFar (secs): " + (timeSoFar() / SECS);
    }
    
    public void printlnTimeSoFar() {
        if (PRINT)
            System.out.println(toString());
    }
    
    public static void main(String[] args) {
        //use case
        
        Timer.PRINT = true; //globally should timers be printed, similar idea to ndebug
        Timer looptimer = new Timer("looptimer");
        
        for (int i = 0; i < 1000000; i++) {
            
        }
        
        looptimer.printlnTimeSoFar();
    }
}
