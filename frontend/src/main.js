import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import VueLogger from 'vuejs-logger';

const isProduction = process.env.NODE_ENV === 'production';
const options = {
  isEnabled: true,
  logLevel : isProduction ? 'error' : 'debug',
  stringifyArguments : false,
  showLogLevel : true,
  showMethodName : true,
  separator: '|',
  showConsoleColors: true
};
Vue.use(VueLogger, options);

Vue.config.productionTip = false;

Vue.use(ElementUI);

const TopMenu = () => import('@/components/TopMenu');
Vue.component('top-menu', TopMenu);

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app');



