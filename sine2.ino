#include "waveData2.h"
#include "TM1637.h"
#define CLK 3
#define DIO 2
TM1637 tm1637(CLK,DIO);
int8_t displayData[4]={1,1,0,1};
unsigned int wholePeriodNum = 512;
unsigned int i, step,k=1;
unsigned int signalOutPWM = 9;                  
unsigned int delaytime = 1950;                 
int value1,value2,value3,k1=1;
int mode=2;
int butten1 = 8;
int butten2 = 13;
int butten3 = 10;
void setup()
 {  
  tm1637.set();
  tm1637.init();
  pinMode(signalOutPWM, OUTPUT);
  pinMode(butten1,INPUT_PULLUP);
  pinMode(butten2,INPUT_PULLUP);
  pinMode(butten3,INPUT_PULLUP);
  step = 1;         
  Serial.begin(9600);
  tm1637.display(displayData);
}
void loop()
 {
  i %= wholePeriodNum;
  if(mode==2){analogWrite(signalOutPWM, sinData[i]*k1/4);}
  else if(mode==1){analogWrite(signalOutPWM, triangularData[i]*k1/4);}
  else if(mode==3){analogWrite(signalOutPWM, squareData[i]*k1/4);}
  i = i + step;
  delayMicroseconds(delaytime);
  value1=digitalRead(8);
  value2=digitalRead(13);
  value3=digitalRead(10);
  if(value1==LOW){k=k+1;
    if(k==21){k=1;step=0;}
    step=step+1;
    displayData[2]=k/10;
    displayData[3]=k%10;
    tm1637.display(displayData);
    delay(500);}
  if(value2==LOW){k1 = k1 + 1;
    if(k1==5){k1=1; }displayData[1]=k1;
    tm1637.display(displayData);
    delay(500);}
  if(value3==LOW){
    mode = mode + 1;
    if(mode==4){mode=1; }
    displayData[0]=mode;
    tm1637.display(displayData);
    delay(500);
  }
}