export const date_str  = {
    now() {
        let d = new Date();
        return  d.getFullYear() + "_" + ("0"+(d.getMonth()+1)).slice(-2) + "_" + ("0" + d.getDate()).slice(-2) + "_" +
            ("0" + d.getHours()).slice(-2) + "_" + ("0" + d.getMinutes()).slice(-2);
    },
    str() {
        let d = new Date();
        return `${d.toDateString()} ${d.toLocaleTimeString()}`;
    }
};

