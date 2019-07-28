// See https://github.com/dialogflow/dialogflow-fulfillment-nodejs
// for Dialogflow fulfillment library docs, samples, and to report issues
'use strict';
 
 const functions = require('firebase-functions');
 const {google} = require('googleapis');
 const {WebhookClient} = require('dialogflow-fulfillment');

process.env.DEBUG = 'dialogflow:debug'; // enables lib debugging statements
 
exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  const agent = new WebhookClient({ request, response });

  const parameters = request.body.queryResult.parameters;
  console.log('Dialogflow Request headers: ' + JSON.stringify(request.headers));
  console.log('Dialogflow Request body: ' + JSON.stringify(request.body));
 
  function welcome(agent) {
    agent.add(`Welcome to my agent!`);
  }
 
  function fallback(agent) {
    agent.add(`I didn't understand`);
    agent.add(`I'm sorry, can you try again?`);
  }
  function test_ans(agent){
    if(parameters.test_entity=="테스트"){
      agent.add("테스트 받아치기!");
    }
  }
  function time_alert(agent){
    var Date_ = new Date();
    var hours = String(Date_.getHours()+1);
    var minites = String(Date_.getMinutes());
    //var time_ = `현재 시간은 {$}시 {$}분 입니다.`;
    var date_ =`오늘은 ${Date_.getMonth()+1}월 {$Date_.getMonth()}일 입니다.`;

    var message_to_agent = "";
    if(parameters.time_now=="현재시간"){
     agent.add(`현재 시간은 ${Date_.getHours()}시 ${Date_.getMinutes()}분 입니다.`);
    }
  }
   function set_appointment(agent) {
     const [일정, 날짜] =
                [agent.parameters['tasks'], agent.parameters['number'], agent.parameters['pizza_size']];
     let missingSlots = [];
     if (!피자타입) { missingSlots.push('피자타입'); }
     if (!피자개수) { missingSlots.push('피자개수'); }
     if (!피자사이즈) { missingSlots.push('피자사이즈'); }

     if (missingSlots.length === 1){
        agent.add(`${missingSlots[0]} 정보가 필요해요`);
      }
      else if (missingSlots.length === 2){
         agent.add(`${missingSlots[0]},  ${missingSlots[1]} 정보가 없는 것 같아요!`);
     }
     else if (missingSlots.length === 3){
         agent.add(` ${missingSlots[0]}, ${missingSlots[1]}, ${missingSlots[2]} 정보를 알려주세요!`);
     } else {
       agent.add(`${피자타입}, ${피자개수}판  ${피자사이즈} 사이즈를 선택하셨군요! 이제 주소를 알려주세요!`);
     }
   }

  function check_day(agent){
    var day_dic = {
      1:"월요일"
      ,2:"화요일"
      ,3:"수요일"
      ,4:"목요일"
      ,5:"금요일"
      ,6:"토요일"
      ,7:"일요일"
    };
    if(parameters.day_entity=="오늘요일"){
      var Date_ = new Date();
       agent.add(`오늘은 ${day_dic[Date_.getDay()]}입니다.`);
    }else if(parameters.day_entity=="내일요일"){
      var tor = Date_.getDay()+1;
      if(tor>7){
        tor=1;
      }
      agent.add(`내일은 ${day_dic[tor]}입니다.`);
    }
    if(parameters.day_entity=="어제요일"){
      var tor = Date_.getDay()-1;
      if(tor<1){
        tor=7;
      }
      agent.add(`내일은 ${day_dic[tor]}입니다.`);

    }

  }
  
  let intentMap = new Map();
  intentMap.set('day_info',check_day);
  intentMap.set('time_info',time_alert);
  intentMap.set('test_intent',test_ans);
  // intentMap.set('your intent name here', yourFunctionHandler);
  // intentMap.set('your intent name here', googleAssistantHandler);
  agent.handleRequest(intentMap);
});
