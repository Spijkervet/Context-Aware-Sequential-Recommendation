cd $1
tail -$3 user-item.lst > te_user-item.lst
tail -$3 user-item-delta-time.lst > te_user-item-delta-time.lst
tail -$3 user-item-accumulate-time.lst > te_user-item-accumulate-time.lst
head -$2 user-item.lst > tr_user-item.lst
head -$2 user-item-delta-time.lst > tr_user-item-delta-time.lst
head -$2 user-item-accumulate-time.lst > tr_user-item-accumulate-time.lst
