import axios from 'axios'
import ulog from 'ulog'
import * as FileSaver from 'file-saver'
import {MessageBox} from "element-ui";

export const http = {
    base: 'http://localhost:8180',
    axiosInstance: axios.create(),
    log: ulog('http'),

    generalErrorHandler(e) {
        let message = e.message;
        if (e.response !== undefined && e.response.data !== undefined) {
            message = e.response.data;
        }
        MessageBox.alert(message, {
            title: "Error",
            type: "error",
            confirmButtonText: 'Close',
        })
    },

    get(url) {
        return this.axiosInstance.get(this.base + url)
    },

    post(url, data) {
        return this.axiosInstance.post(this.base + url, data)
    },

    put(url, data) {
        return this.axiosInstance.put(this.base + url, data)
    },

    getDataList() {
        return this.get('/data/list')
    },

    getClassifierList() {
        return this.get('/classifier/list')
    },

    saveAsFile (response, name) {
        let blob = new Blob([response.data]);
        FileSaver.saveAs(blob, name)
    },

    postAsync(url, data, periodic, on_error, on_done) {
        this.axiosInstance.post(this.base + url, data)
            .then(r => {
                this.startPeriodic(r.data, periodic, on_error, on_done);
            })
            .catch(e => {
                if (on_error !== undefined) {
                    on_error(e);
                } else {
                    this.log.error({e});
                }
            });
    },
    startPeriodic(job_id, callback, on_error, on_done) {
        let handler = () => {
            this.get(`/job/${job_id}/status`)
                .then(r => {
                    callback(r.data.status, job_id);
                    if (r.data.status === "Pending" || r.data.status.Progress !== undefined) {
                        setTimeout(handler, 1000);
                    } else if (on_done !== undefined) {
                        if (r.data.status.Ok !== undefined) {
                            on_done(r.data.status.Ok);
                        } else {
                            on_done(r.data);
                        }
                    }
                })
                .catch(e => {
                    if (on_error !== undefined) {
                        on_error(e);
                    } else {
                        this.log.error({e});
                    }
                })
        };
        handler();
    }
};
