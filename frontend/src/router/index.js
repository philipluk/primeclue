import Vue from 'vue'
import VueRouter from 'vue-router'
import DataWindow from "../components/data/DataWindow";
import ClassifiersWindow from "../components/classifier/ClassifiersWindow";
import ClassifierCreate from "../components/classifier/ClassifierCreate";
import DataImport from "../components/data/DataImport";
import Classification from "../components/classification/Classification";

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    redirect: '/data-window'
  },
  {
    path: '/data-window',
    name: 'data-window',
    component: DataWindow
  },
  {
    path: '/data-import',
    name: 'data-import',
    component: DataImport
  },
  {
    path: '/classifiers-window',
    name: 'classifiers-window',
    component: ClassifiersWindow
  },
  {
    path: '/classifier-create',
    name: 'classifier-create',
    component: ClassifierCreate,
    props: true,
  },
  {
    path: '/classification',
    name: 'classification',
    component: Classification,
    props: true,
  },
  {
    path: '/about',
    name: 'about',
    component: () => import(/* webpackChunkName: "about" */ '../views/About.vue')
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
