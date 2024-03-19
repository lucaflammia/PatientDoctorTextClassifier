import http from 'k6/http';
import { check, group, sleep } from "k6";

export let options = {
  stages: [
    { duration: '10m', target: 100 }, // simulate ramp-up of traffic 
    { duration: '30m', target: 100 }, // stay at a certain value of traffic
    { duration: '5m', target: 0 }, // ramp-down
  ]
};

export default function () {
  group("v1 API testing", () => {
    var url = 'https://66qmcbg0y3.execute-api.eu-west-1.amazonaws.com/text-classification/patient-doctor-text-classifier';
    var payload = JSON.stringify({
      "text": "I prescribe some drugs I force you to stay at home for a week."
    });
    var params = {
      headers: {
        'Content-Type': 'application/json',
      },
    };
    group('Make predictions about input texts', () => {
      const res = http.post(url, payload, params);
      check(res, {
        "is status 200": (r) => r.status == 200
      });
    });
  });
  sleep(1)
}